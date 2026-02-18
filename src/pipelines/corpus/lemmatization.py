import argparse
import hashlib
import logging
import signal
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List, Optional, Dict

import re
from dotenv import load_dotenv

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.io.cloud import DefaultUploadCallback, S3Mapping
from cclang.io.exceptions import CloudStorageError, LocalStorageError
from cclang.core.storage import StorageManager, CloudConfig
from cclang.core.manifest import ManifestStore
from cclang.io.fs import LocalConfig, ensure_relative, get_shard_relative
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.schemas import (
    DocLemma,
    DocTok,
    LemmatizeTask,
    LemmaToken,
    LemmatizeManifestRecord,
    ProcessingStatus, ProcessedPdfStatus,
)
from cclang.models.tasks_queue import TaskQueue
from cclang.corpus.lemmatizers.apertium_mar import lemmatize_marathi_token, batch_lemmatize_marathi

DEVANAGARI_RANGE = r"\u0900-\u097F"
DEVANAGARI_DIGITS = r"\u0966-\u096F"


@dataclass
class LemmaStatus:
    doc_id: str
    method: str
    lemma_status: ProcessingStatus
    lemma_sha: Optional[str] = None
    lemma_path: Optional[str] = None


_callback_logger = get_logger(__name__)


class UploadArtifactCallback(DefaultUploadCallback):
    def __init__(self, manifest_store: ManifestStore, record_on_success: LemmatizeManifestRecord,
                 database: DocStateStore, db_update_info: LemmaStatus):
        super().__init__()
        self.manifest_store = manifest_store
        self.record = record_on_success
        self.database = database
        self.db_update_info = db_update_info

    def on_success(self, mapping: S3Mapping, extra: Optional[Dict] = None) -> None:
        super().on_success(mapping, extra)
        self.manifest_store.mark(self.record)
        self.database.upsert_lemma_result(
            doc_id=self.db_update_info.doc_id,
            method=self.db_update_info.method,
            status=self.db_update_info.lemma_status,
            artefact_id=self.db_update_info.lemma_sha,
            path=self.db_update_info.lemma_path,
            ts=datetime.now().isoformat() + 'Z'
        )

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
class LemmatizeWorkerResult:
    doc_id: str
    tokenize_path: str
    status: ProcessedPdfStatus
    temp_result_path: Optional[Path]
    error: Optional[str] = None


class Counters:
    """Thread-safe counters for lemmatization statistics."""
    def __init__(self):
        self._lock = threading.Lock()
        self.num_oov: int = 0
        self.num_ambiguous: int = 0
        self.num_weird: int = 0

    def bump(self, name: str, logger: BoundLogger) -> None:
        with self._lock:
            current = getattr(self, name) + 1
            setattr(self, name, current)
        if current % 500 == 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug("counter threshold", extra={"counter": name, "value": current})


def _lemmatize_doc(doc: DocTok, log: BoundLogger, global_counters: Counters | None) -> DocLemma:
    """
    Instead of calling apertium once per token, collects all unique tokens
    and processes them in batches.
    """
    local_counters = Counters()

    # Step 1: Collect unique non-weird tokens
    unique_tokens: set[str] = set()
    for sentence in doc.sentences:
        for tok in sentence:
            if not _is_weird_token(tok):
                unique_tokens.add(tok)

    # Step 2: Batch lemmatize all unique tokens
    token_cache = batch_lemmatize_marathi(list(unique_tokens), batch_size=50)

    # Step 3: Build output sentences using the cache
    sentences: List[List[LemmaToken]] = []
    for sentence in doc.sentences:
        lemma_sentence: List[LemmaToken] = []
        for tok in sentence:
            if _is_weird_token(tok):
                if global_counters is not None:
                    global_counters.bump("num_weird", log)
                local_counters.num_weird += 1
                continue

            lemma_tok = token_cache.get(tok)
            if lemma_tok is None:
                # Fallback (shouldn't happen)
                lemma_tok = lemmatize_marathi_token(tok)

            if lemma_tok.is_oov:
                if global_counters is not None:
                    global_counters.bump("num_oov", log)
                local_counters.num_oov += 1
            if lemma_tok.is_ambiguous:
                if global_counters is not None:
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
        stop_event: threading.Event | None = None,
):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    output_base_rel = ensure_relative(Path(output_base_path), data_path, "lemma_output_base_path")
    output_base_abs = data_path / output_base_rel
    output_base_abs.mkdir(parents=True, exist_ok=True)

    db_conn = get_conn(database_dsn)

    s3_cfg = load_s3_config()
    loc_cfg = LocalConfig(
        base_path=data_path,
        save_local=True,
        cache_files=True,
        temp_base=data_path / "temp/lemmatization",
    )
    cloud_cfg = CloudConfig(enable=s3_cfg.enable,
                            base_path=Path("data"),
                            s3_config=s3_cfg,
                            max_upload_threads=2,
                            max_pool_connections=48)
    storage_manager = StorageManager(loc_cfg, cloud_cfg)

    manifest = storage_manager.register_manifest(manifest_path, LemmatizeManifestRecord)
    doc_store = DocStateStore(db_conn)
    lemma_method = "apertium-mar-morph"

    try:
        log.info(
            "pipeline target tasks",
            extra={"lemma_status": target_status, "method": lemma_method, "limit": max_count},
        )
        tasks_filter = doc_store.get_lemma_tasks(target_status, method=lemma_method, limit=max_count)
        task_queue: TaskQueue[LemmatizeTask] = TaskQueue(logger=log.bind(service="TaskQueue"))
        for task in tasks_filter:
            task_queue.put(task)

        result_queue: Queue[LemmatizeWorkerResult] = Queue()
        global_counters = Counters() if log.isEnabledFor(logging.DEBUG) else None

        log.info("pipeline start", extra={"tasks": task_queue.qsize()})

        def lemma_worker(worker_log: BoundLogger):
            while True:
                if stop_event is not None and stop_event.is_set():
                    worker_log.info("stop signal received, exiting worker")
                    return
                try:
                    task = task_queue.get(timeout=2)
                except Empty:
                    # Check if we should stop or if queue is exhausted
                    if stop_event is not None and stop_event.is_set():
                        return
                    if task_queue.qsize() == 0 and task_queue.unfinished_tasks == 0:
                        return  # All tasks done
                    continue

                tokenized_path_raw = str(task.tokenize_path).replace("\\", "/")
                normalized_tok_path = ensure_relative(Path(tokenized_path_raw), data_path, "tokenize_path")
                worker_result: LemmatizeWorkerResult = LemmatizeWorkerResult(
                    doc_id=task.doc_id,
                    tokenize_path=str(normalized_tok_path),
                    status=ProcessedPdfStatus.SKIPPED,
                    temp_result_path=None)

                temp_dist = storage_manager.create_temp_file(".lemma.json")
                keep_file = False
                try:
                    with storage_manager.open(normalized_tok_path, mode="r", encoding="utf-8") as stream:
                        doc_tok = DocTok.model_validate_json(stream.read())

                    worker_log.info('Started lemmatization', extra={"doc_id": task.doc_id})
                    doc_lemma = _lemmatize_doc(doc_tok, worker_log, global_counters)
                    worker_log.info('Finish lemmatization', extra={"doc_id": task.doc_id})

                    lemma_json = doc_lemma.model_dump_json(ensure_ascii=False)
                    with open(temp_dist, mode="w", encoding="utf-8") as stream:
                        stream.write(lemma_json)

                    worker_log.info('Lemmatization was successful. Saved local', extra={"doc_id": task.doc_id})
                    worker_result = LemmatizeWorkerResult(
                        doc_id=task.doc_id,
                        tokenize_path=str(normalized_tok_path),
                        status=ProcessedPdfStatus.OK,
                        temp_result_path=temp_dist,
                    )
                    keep_file = True
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

                    worker_result = LemmatizeWorkerResult(
                        doc_id=task.doc_id,
                        tokenize_path=str(normalized_tok_path),
                        status=ProcessedPdfStatus.ERROR,
                        temp_result_path=None,
                        error=str(exc),
                    )
                finally:
                    task_queue.task_done()
                    if not keep_file and temp_dist is not None:
                        try:
                            temp_dist.unlink()
                        except Exception:  # noqa : BLE001
                            worker_log.warning("failed to cleanup temp file", extra={"path": str(temp_dist)})

                result_queue.put(worker_result)

        work_threads = [
            threading.Thread(target=lemma_worker, args=(log.bind(thread_name=f"lemma_worker_{ind}"),))
            for ind in range(8)
        ]
        for work_thread in work_threads:
            work_thread.start()

        stopped = False
        while any(thread.is_alive() for thread in work_threads) or not result_queue.empty():
            if stop_event is not None and stop_event.is_set() and not stopped:
                log.warning("stop signal received, finishing current tasks...")
                stopped = True
            try:
                result = result_queue.get(timeout=2)
                lemma_path: Optional[Path] = None
                lemma_sha: Optional[str] = None
                lemma_status: ProcessingStatus = ProcessingStatus.NOT_PROCESSED

                if result.status == ProcessedPdfStatus.OK:
                    lemma_path = output_base_rel / get_shard_relative(result.doc_id, ".lemma.json")
                    lemma_sha = hashlib.sha256(result.temp_result_path.read_bytes()).hexdigest()
                    lemma_status = ProcessingStatus.OK

                    manifest_record = LemmatizeManifestRecord(
                        pdf_sha=result.doc_id,
                        tokenize_path=result.tokenize_path,
                        status=result.status,
                        lemma_path=str(lemma_path),
                        lemma_sha=lemma_sha
                    )

                    if storage_manager.save_cloud_enabled():
                        upload_callback = UploadArtifactCallback(
                            manifest, manifest_record, doc_store,
                            LemmaStatus(result.doc_id, lemma_method,
                                        lemma_status, lemma_sha, str(lemma_path)))
                        storage_manager.finalize_artifact(result.temp_result_path, lemma_path,
                                                          blocking=False, callback=upload_callback)
                    else:
                        storage_manager.finalize_artifact(result.temp_result_path, lemma_path,
                                                          blocking=True)
                        manifest.mark(manifest_record)
                        doc_store.upsert_lemma_result(
                            doc_id=result.doc_id,
                            method=lemma_method,
                            status=lemma_status,
                            artefact_id=lemma_sha,
                            path=str(lemma_path),
                            ts=datetime.now().isoformat() + 'Z'
                        )
                elif result.status == ProcessedPdfStatus.ERROR:
                    lemma_status = ProcessingStatus.ERROR

                    manifest_record = LemmatizeManifestRecord(
                        pdf_sha=result.doc_id,
                        tokenize_path=result.tokenize_path,
                        status=result.status,
                        lemma_path=None,
                        lemma_sha=None,
                        error=result.error,
                    )
                    manifest.mark(manifest_record)

                    doc_store.upsert_lemma_result(
                        doc_id=result.doc_id,
                        method=lemma_method,
                        status=lemma_status,
                        artefact_id=lemma_sha,
                        path=str(lemma_path) if lemma_path else None,
                        ts=datetime.now().isoformat() + 'Z'
                    )
                elif result.status == ProcessedPdfStatus.SKIPPED:
                    # SKIPPED should not reach here in normal operation
                    log.warning("unexpected SKIPPED status received", extra={"doc_id": result.doc_id})
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
            extra=({
                       "num_oov": global_counters.num_oov,
                       "num_ambiguous": global_counters.num_ambiguous,
                       "num_weird_token": global_counters.num_weird,
                   } if global_counters is not None else {}),
        )
    finally:
        manifest.flush()
        storage_manager.close()  # Wait for uploads + callbacks first (they write to doc_store)
        doc_store.close()        # Now safe to close DB connection
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
    arg_parser.add_argument("--manifest-path", "--manifest", required=False, help="path to manifest", type=Path,
                            default=None)
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
    arg_parser.add_argument("--head", "-head", required=False, default=None,
                            help="Max count of links that will be processed", type=int)
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
        ProcessingStatus.NOT_PROCESSED if args.target_status in ("none", "pending") else ProcessingStatus(
            args.target_status)
    )

    # Set up graceful shutdown on SIGINT/SIGTERM
    stop_event = threading.Event()

    def _handle_signal(sig, _frame):
        if stop_event.is_set():
            return  # Already stopping
        log.warning("received stop signal, initiating graceful shutdown...", extra={"signal": sig})
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except (OSError, ValueError):
                pass  # Signal not supported on this platform

    try:
        return run_pipeline(
            output_base_path=args.output_base_path,
            database_dsn=database_dsn,
            manifest_path=manifest_path,
            log=log,
            target_status=target_status,
            max_count=max_count,
            data_path=data_path,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        log.warning("interrupted by user")
    except Exception:
        log.exception("unexpected error")
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
