import argparse
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List

import re
from dotenv import load_dotenv

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.ingest.exceptions import CloudStorageError, LocalStorageError
from cclang.ingest.fs import CloudConfig, FileManager, LocalConfig, ensure_relative, get_shard_relative
from cclang.io.db import get_conn
from cclang.io.manifest import ManifestStore
from cclang.io.pdf_state_store import PdfStateStore
from cclang.io.schemas import (
    DocLemma,
    DocTok,
    LemmaToken,
    LemmatizeManifestRecord,
    ProcessingStatus,
)
from cclang.models.tasks_queue import TaskQueue


def lemmatize_marathi_token(token: str) -> LemmaToken:
    """
    Placeholder: assumed to be implemented elsewhere.
    Should return LemmaToken with fields filled (lemma, pos, analyses, is_oov, is_ambiguous).
    """
    raise NotImplementedError


DEVANAGARI_RANGE = r"\u0900-\u097F"
DEVANAGARI_DIGITS = r"\u0966-\u096F"


def _is_weird_token(token: str) -> bool:
    """
    Return True if token is not Marathi letters or digits (Devanagari or ASCII).
    """
    if re.fullmatch(rf"[{DEVANAGARI_DIGITS}0-9]+", token):
        return False
    if re.fullmatch(rf"[{DEVANAGARI_RANGE}]+", token):
        return False
    return True


@dataclass
class Counters:
    num_oov: int = 0
    num_ambiguous: int = 0
    num_weird: int = 0

    def bump(self, name: str, logger: BoundLogger) -> None:
        current = getattr(self, name) + 1
        setattr(self, name, current)
        if current % 500 == 0 and logger.isEnabledFor(logger.logger.DEBUG):
            logger.warning("counter threshold", extra={"counter": name, "value": current})


def _lemmatize_doc(doc: DocTok, log: BoundLogger, global_counters: Counters) -> DocLemma:
    sentences: List[List[LemmaToken]] = []
    local_counters = Counters()

    for sentence in doc.sentences:
        lemma_sentence: List[LemmaToken] = []
        for tok in sentence:
            if _is_weird_token(tok):
                global_counters.bump("num_weird", log)
                local_counters.num_weird += 1
                continue

            lemma_tok = lemmatize_marathi_token(tok)

            if lemma_tok.is_oov:
                global_counters.bump("num_oov", log)
                local_counters.num_oov += 1
            if lemma_tok.is_ambiguous:
                global_counters.bump("num_ambiguous", log)
                local_counters.num_ambiguous += 1

            lemma_sentence.append(lemma_tok)
        sentences.append(lemma_sentence)

    meta = {
        "source_tok_sha": doc.id,
        "lemmatizer": "apertium-mar-morph",
        "stats": {
            "num_sentences": len(sentences),
            "num_tokens": sum(len(s) for s in sentences),
            "num_oov": local_counters.num_oov,
            "num_ambiguous": local_counters.num_ambiguous,
            "num_weird_token": local_counters.num_weird,
        },
    }
    return DocLemma(id=doc.id, lang=doc.lang, sentences=sentences, meta=meta)


def run_pipeline(
    output_base_path: Path,
    database_dsn: str | None,
    manifest_path: Path,
    log: BoundLogger,
    target_status: ProcessingStatus | None = ProcessingStatus.NOT_PROCESSED,
    max_count: int | None = None,
    data_path: Path = Path("data"),
):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    output_base_rel = ensure_relative(Path(output_base_path), data_path, "lemma_output_base_path")
    output_base_abs = data_path / output_base_rel
    output_base_abs.mkdir(parents=True, exist_ok=True)

    db_conn = get_conn(database_dsn)

    s3_cfg = load_s3_config()
    file_manager = FileManager(
        local_cfg=LocalConfig(
            base_path=data_path,
            save_local=True,
            cache_files=True,
            temp_base=data_path / "temp/lemmatization",
        ),
        cloud_cfg=CloudConfig(enable=s3_cfg.enable, base_path=Path("data"), s3_config=s3_cfg, max_upload_threads=2),
    )

    manifest = ManifestStore(manifest_path, LemmatizeManifestRecord, file_manager)
    pdf_states = PdfStateStore(db_conn)

    log.info(
        "pipeline target tasks",
        extra={"tokenize_status": ProcessingStatus.OK, "lemma_status": target_status, "limit": max_count},
    )
    tasks_filter = pdf_states.filter_by_status(text_status=ProcessingStatus.OK,
                                               tokenize_status=ProcessingStatus.OK,
                                               lemma_status=target_status)
    task_queue: TaskQueue = TaskQueue(logger=log.bind(service="TaskQueue"))
    for task in tasks_filter:
        task_queue.put(task)
        if max_count is not None and task_queue.qsize() >= max_count:
            break

    result_queue: Queue[LemmatizeManifestRecord] = Queue()
    global_counters = Counters()

    log.info("pipeline start", extra={"tasks": task_queue.qsize()})

    def lemma_worker(worker_log: BoundLogger):
        while True:
            try:
                task = task_queue.get(timeout=10)
            except Empty:
                return

            tokenized_path_raw = str(task.tokenize_path).replace("\\", "/")
            normalized_tok_path = ensure_relative(Path(tokenized_path_raw), data_path, "tokenize_path")
            result: LemmatizeManifestRecord

            try:
                with file_manager.load_data(normalized_tok_path, mode="r") as stream:
                    doc_tok = DocTok.model_validate_json(stream.read())

                doc_lemma = _lemmatize_doc(doc_tok, worker_log, global_counters)
                shard_relative_result = output_base_rel / get_shard_relative(task.pdf_sha, ".lemma.json")

                lemma_json = doc_lemma.model_dump_json(ensure_ascii=False)
                with file_manager.load_data(shard_relative_result, mode="w") as stream:
                    stream.write(lemma_json)

                lemma_sha = hashlib.sha256(lemma_json.encode()).hexdigest()
                result = LemmatizeManifestRecord(
                    pdf_sha=task.pdf_sha,
                    tokenize_path=str(normalized_tok_path),
                    lemma_path=str(shard_relative_result),
                    lemma_sha=lemma_sha,
                    status=LemmatizeManifestRecord.STATUS_OK,
                )
            except Exception as exc:
                if isinstance(exc, LocalStorageError):
                    worker_log.exception(
                        "local storage error during lemmatization",
                        extra={"tokenize-file": normalized_tok_path},
                        exc=exc,
                    )
                elif isinstance(exc, CloudStorageError):
                    worker_log.exception(
                        "cloud storage error during lemmatization",
                        extra={"tokenize-file": normalized_tok_path},
                        exc=exc,
                    )
                else:
                    worker_log.exception(
                        "unexpected error during lemmatization",
                        extra={"tokenize-file": normalized_tok_path},
                        exc=exc,
                    )

                result = LemmatizeManifestRecord(
                    pdf_sha=task.pdf_sha,
                    tokenize_path=str(normalized_tok_path),
                    lemma_path=None,
                    lemma_sha=None,
                    status=LemmatizeManifestRecord.STATUS_ERROR,
                    error=str(exc),
                )
            finally:
                task_queue.task_done()

            result_queue.put(result)

    work_threads = [
        threading.Thread(target=lemma_worker, args=(log.bind(thread_name=f"lemma_worker_{ind}"),))
        for ind in range(8)
    ]
    for work_thread in work_threads:
        work_thread.start()

    while any(thread.is_alive() for thread in work_threads) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=10)
            manifest.mark(result)

            if result.status in (LemmatizeManifestRecord.STATUS_OK, LemmatizeManifestRecord.STATUS_SKIPPED):
                lemma_status = ProcessingStatus.OK
            elif result.status == LemmatizeManifestRecord.STATUS_ERROR:
                lemma_status = ProcessingStatus.ERROR
            else:
                raise RuntimeError("unexpected lemmatizer behavior")

            pdf_states.update_lemma_status(
                pdf_sha=result.pdf_sha,
                lemma_status=lemma_status,
                lemma_sha=result.lemma_sha,
                ts=result.ts,
            )
        except Empty:
            if task_queue.unfinished_tasks == 0 and result_queue.empty():
                break
            continue

    for thread in work_threads:
        thread.join()

    task_queue.join()
    task_queue.stop_logging()

    log.info(
        "lemmatization completed",
        extra={
            "num_oov": global_counters.num_oov,
            "num_ambiguous": global_counters.num_ambiguous,
            "num_weird_token": global_counters.num_weird,
        },
    )
    manifest.flush()
    pdf_states.close()
    file_manager.close()
    log.info("files closed")


def main(argv: Iterable[str] | None = None) -> None:
    load_dotenv()
    arg_parser = argparse.ArgumentParser(description="Lemmatize tokenized Marathi text")

    arg_parser.add_argument("--output-base-path", "-output", required=True, help="output base path", type=Path)
    arg_parser.add_argument(
        "--database-dsn",
        "--database-path",
        "--database-url",
        "-db",
        dest="database_dsn",
        required=False,
        help="PostgreSQL DSN; if omitted, falls back to CCLANG_DB_DSN env var",
        type=str,
        default=None,
    )
    arg_parser.add_argument("--manifest-path", "--manifest", required=False, help="path to manifest", type=Path, default=None)
    arg_parser.add_argument(
        "--target-status",
        "-st",
        choices=["pending", "none", ProcessingStatus.OK.value, ProcessingStatus.ERROR.value],
        default="pending",
        help="Which lemma_status to process (pending/none = not processed yet)",
    )
    arg_parser.add_argument("--log-format", "--log-fmt", choices=["console", "json"], default="console")
    arg_parser.add_argument("--log-level", "--log-lvl", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    arg_parser.add_argument("--log-file", default=None, help="File for logs (JSONL)")
    arg_parser.add_argument("--head", "-head", required=False, default=None, help="Max count of links that will be processed", type=int)
    arg_parser.add_argument(
        "--data-path",
        "-data",
        required=False,
        default=None,
        type=Path,
        help="Root directory for pipeline artifacts (defaults to ./data)",
    )

    args = arg_parser.parse_args(argv)

    manifest_path = args.manifest_path or Path("manifests/pl_lemmatization.jsonl")
    database_dsn = args.database_dsn
    data_path = args.data_path or Path("data")

    log_file = Path(args.log_file or Path("data/logs/corpus/lemmatization.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(
        fmt=args.log_format,
        level=args.log_level,
        file=str(log_file),
    )
    log = get_logger("root").bind(pipeline="lemmatization")
    max_count = args.head if args.head is not None else None
    target_status = (
        ProcessingStatus.NOT_PROCESSED if args.target_status in ("none", "pending") else ProcessingStatus(args.target_status)
    )
    try:
        return run_pipeline(
            output_base_path=args.output_base_path,
            database_dsn=database_dsn,
            manifest_path=manifest_path,
            log=log,
            target_status=target_status,
            max_count=max_count,
            data_path=data_path,
        )
    except Exception:
        log.exception("unexpected error")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
