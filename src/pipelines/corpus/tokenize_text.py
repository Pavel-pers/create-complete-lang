import argparse
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import List, Iterable
from dotenv import load_dotenv
import re
import string
import unicodedata
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
import hashlib

from cclang.config.s3 import load_s3_config
from cclang.io.db import get_conn
from cclang.core.manifest import ManifestStore
from cclang.io.pdf_state_store import PdfStateStore
from cclang.io.schemas import DocRaw, DocTok, ProcessingStatus, TokenizeManifestRecord
from cclang.common.logx import BoundLogger, setup_logging, get_logger
from cclang.io.fs import FileManager, ensure_relative, LocalConfig, CloudConfig, get_shard_relative
from cclang.io.exceptions import LocalStorageError, CloudStorageError
from cclang.models.tasks_queue import TaskQueue


def _clean_marathi_text(text: str) -> str:
    if not hasattr(_clean_marathi_text, "_cfg"):
        lang_range = "\u0900-\u097F"  # диапазон деванагари
        extra_punct = "।“”‘’—–…«»"
        punctuation_range = string.punctuation + extra_punct
        digit_range = "0-9"
        whitespace_range = r"\s"

        bad_symbols_pattern = (
            f"[^{lang_range}{digit_range}{re.escape(punctuation_range)}{whitespace_range}]"
        )

        cfg = {"bad_symbols_re": re.compile(bad_symbols_pattern),
               "url_re": re.compile(r"https?://\S+|www\.\S+"),
               "multispace_re": re.compile(r"\s+"),
               "multipunct_re": re.compile(r"([?!.,।]){2,}")
               }

        devanagari_digits = "०१२३४५६७८९"
        ascii_digits = "0123456789"
        cfg["digit_trans"] = str.maketrans(dict(zip(devanagari_digits, ascii_digits)))
        _clean_marathi_text._cfg = cfg

    cfg = _clean_marathi_text._cfg

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = cfg["url_re"].sub(" ", text)
    text = text.translate(cfg["digit_trans"])
    text = cfg["bad_symbols_re"].sub(" ", text)
    text = cfg["multipunct_re"].sub(r"\1", text)
    text = cfg["multispace_re"].sub(" ", text).strip()
    return text


def _iter_text_pages(book_path: Path, fm: FileManager):
    with fm.ensure_file(book_path, mode='r') as stream:
        for line in stream:
            yield DocRaw.model_validate_json(line).text


def _tokenize_text(text: str) -> List[List[str]]:
    text = _clean_marathi_text(text)
    sentences = sentence_tokenize.sentence_split(text, lang='mr')
    sentences_tokens = map(lambda ss: indic_tokenize.trivial_tokenize(ss, lang='mr'), sentences)
    return list(sentences_tokens)


def tokenize_text(pdf_path: Path, pdf_sha: str, logger: BoundLogger, fm: FileManager) -> DocTok:
    logger.debug('start tokenize text', extra={"text_path": str(pdf_path)})

    book_text = '\n\n'.join(_iter_text_pages(pdf_path, fm))
    book_tokens = _tokenize_text(book_text)
    logger.debug('finish tokenize text', extra={"text_path": str(pdf_path)})
    return DocTok(id=pdf_sha, lang='mr', sentences=book_tokens,
                  meta={
                      'source_pdf_sha': pdf_sha,
                      'source_pdf_path': str(pdf_path),
                      'tokenizer': 'indic-nlp-library',
                      'stats': {
                          'num_sentences': len(book_tokens),
                          'num_tokens': sum(map(len, book_tokens))
                      }})


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

    output_base_rel = ensure_relative(Path(output_base_path), data_path, "tokenize_output_base_path")
    output_base_abs = data_path / output_base_rel
    output_base_abs.mkdir(parents=True, exist_ok=True)

    db_conn = get_conn(database_dsn)

    s3_cfg = load_s3_config()
    file_manager = FileManager(
        local_cfg=LocalConfig(
            base_path=data_path,
            save_local=True,
            cache_files=True,
            temp_base=data_path / "temp/tokenize_text",
        ),
        cloud_cfg=CloudConfig(enable=s3_cfg.enable, base_path=Path("data"), s3_config=s3_cfg, max_upload_threads=2),
    )

    manifest = ManifestStore(manifest_path, TokenizeManifestRecord, file_manager)
    pdf_states = PdfStateStore(db_conn)

    log.info(f'pipeline target tasks: text_status={ProcessingStatus.OK}, tokenize_status= {target_status}, limit={max_count}')
    tasks_filter = pdf_states.filter_by_status(text_status=ProcessingStatus.OK,
                                               tokenize_status=target_status)
    task_queue = TaskQueue(logger=log.bind(service='TaskQueue'))
    for task in tasks_filter:
        task_queue.put(task)
        if max_count is not None and task_queue.qsize() >= max_count:
            break

    result_queue: Queue[TokenizeManifestRecord]  = Queue()

    log.info(f'pipeline start working on {task_queue.qsize()} tasks')

    def tokenize_worker(worker_log: BoundLogger):
        while True:
            try:
                task = task_queue.get(timeout=10)
            except Empty:
                return

            # Normalize Windows-style backslashes in stored paths to POSIX separators
            text_path_raw = str(task.text_path).replace("\\", "/")
            normalized_text_path = ensure_relative(Path(text_path_raw), data_path, "text_path")
            result: TokenizeManifestRecord

            try:
                tokenized_doc = tokenize_text(normalized_text_path, task.pdf_sha, worker_log, file_manager)
                shard_relative_result = output_base_rel / get_shard_relative(task.pdf_sha, '.tok.json')
                tokenized_json = tokenized_doc.model_dump_json(ensure_ascii=False)
                with file_manager.ensure_file(shard_relative_result, mode='w') as stream:
                    stream.write(tokenized_json)

                tokenize_sha = hashlib.sha256(tokenized_json.encode()).hexdigest()
                result = TokenizeManifestRecord(pdf_sha=task.pdf_sha,
                                                text_path=str(normalized_text_path),
                                                tokenize_path=str(shard_relative_result),
                                                tokenize_sha=tokenize_sha,
                                                status=TokenizeManifestRecord.STATUS_OK,
                                                )
            except Exception as exc:
                if isinstance(exc, LocalStorageError):
                    worker_log.exception('local storage error during tokenization', extra={"text-file": normalized_text_path}, exc=exc)
                elif isinstance(exc, CloudStorageError):
                    worker_log.exception('cloud storage error during tokenization', extra={"text-file": normalized_text_path}, exc=exc)
                else:
                    worker_log.exception("unexpected error during tokenization", extra={"text-file": normalized_text_path}, exc=exc)

                result = TokenizeManifestRecord(pdf_sha=task.pdf_sha,
                                                text_path=str(normalized_text_path),
                                                tokenize_path=None,
                                                tokenize_sha=None,
                                                status=TokenizeManifestRecord.STATUS_ERROR,
                                                error=str(exc))

            finally:
                task_queue.task_done()

            result_queue.put(result)


    work_threads = [
        threading.Thread(target=tokenize_worker, args=(log.bind(thread_name=f'tokenize_worker_{ind}'),))
        for ind in range(16)]
    for work_thread in work_threads:
        work_thread.start()

    while any(thread.is_alive() for thread in work_threads) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=10)
            manifest.mark(result)

            tokenize_status: ProcessingStatus
            if result.status in (TokenizeManifestRecord.STATUS_OK, TokenizeManifestRecord.STATUS_SKIPPED):
                tokenize_status = ProcessingStatus.OK
            elif result.status == TokenizeManifestRecord.STATUS_ERROR:
                tokenize_status = ProcessingStatus.ERROR
            else:
                raise RuntimeError('unexpected tokenizer behavior')

            pdf_states.update_tokenize_status(
                pdf_sha=result.pdf_sha,
                tokenize_status=tokenize_status,
                tokenize_sha=result.tokenize_sha,
                tokenize_path=result.tokenize_path,
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

    log.info("tokenize text completed, closing files")
    manifest.flush()
    pdf_states.close()
    file_manager.close()
    log.info("files closed")

def main(argv: Iterable[str] | None = None) -> None:
    load_dotenv()
    arg_parser = argparse.ArgumentParser(
        description='Tokenize text from raw text files'
    )

    arg_parser.add_argument('--output-base-path', '-output', required=True, help='output base path', type=Path)
    arg_parser.add_argument(
        '--database-dsn',
        '--database-path',
        '--database-url',
        '-db',
        dest='database_dsn',
        required=False,
        help='PostgreSQL DSN; if omitted, falls back to CCLANG_DB_DSN env var',
        type=str,
        default=None,
    )
    arg_parser.add_argument('--manifest-path', '--manifest', required=False, help='path to manifest',
                            type=Path, default=None)
    arg_parser.add_argument('--target-status', '-st',
                            choices=[ProcessingStatus.NOT_PROCESSED.value, ProcessingStatus.OK.value, ProcessingStatus.ERROR.value],
                            default=ProcessingStatus.NOT_PROCESSED.value, help="Which text_status to process (pending = not processed yet)")
    arg_parser.add_argument('--log-format', '--log-fmt', choices=["console", "json"], default="console")
    arg_parser.add_argument('--log-level', '--log-lvl', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    arg_parser.add_argument("--log-file",default=None, help="File for logs (JSONL)")
    arg_parser.add_argument("--head", '-head', required=False, default=None,
                            help="Max count of links that will be proccessed", type=int)
    arg_parser.add_argument(
        '--data-path', '-data',
        required=False,
        default=None,
        type=Path,
        help='Root directory for pipeline artifacts (defaults to ./data)',
    )

    args = arg_parser.parse_args(argv)

    manifest_path = args.manifest_path or Path("manifests/pl_tokenize_text.jsonl")
    database_dsn = args.database_dsn
    data_path = args.data_path or Path("data")

    log_file = Path(args.log_file or Path("data/logs/corpus/tokenize_text.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(
        fmt=args.log_format,
        level=args.log_level,
        file=str(log_file),
    )
    log = get_logger("root").bind(pipeline="tokenize_text")
    max_count = args.head if args.head is not None else None

    target_status = ProcessingStatus(args.target_status)
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
    raise SystemExit(main(sys.argv[1:]))
