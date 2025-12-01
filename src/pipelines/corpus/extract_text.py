import argparse
import sys
import threading
from pathlib import Path
from queue import Queue, Empty
import pytesseract
from dataclasses import dataclass
import pdf2image
from typing import Optional, Iterable

from cclang.common.logx import BoundLogger, setup_logging, get_logger
from cclang.config.s3 import load_s3_config
from cclang.ingest import fs
from cclang.ingest.fs import LocalConfig, CloudConfig
from cclang.io.db import get_conn
from cclang.io.manifest import ManifestStore
from cclang.io.pdf_state_store import PdfStateStore
from cclang.io.schemas import PdfState, ProcessingStatus, ProcessedPdfManifestRecord, DocRaw
from cclang.models.tasks_queue import TaskQueue
from cclang.ingest.fs import FileManager


def _ensure_relative(path: Path, base: Path, label: str) -> Path:
    """Convert absolute path to relative to base; accept already-relative paths."""
    if path.is_absolute():
        try:
            return path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside data base path {base}") from exc
    if path.parts and path.parts[0] == base.name:
        return Path(*path.parts[1:])
    return path


def _ensure_absolute(path: Path, base: Path) -> Path:
    """Return absolute path rooted at base when input is relative."""
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == base.name:
        path = Path(*path.parts[1:])
    return base / path


def extract_text_from_page(page):
    text = pytesseract.image_to_string(
        page,
        lang='mar',
        config=r'--oem 1 --psm 6'
    )
    return text


@dataclass
class ExtractedTextResult:
    tmp_file: Optional[Path]
    text_sha: Optional[str]
    error: Optional[str] = None


def extract_text_from_pdf_to_temp(pdf_path: Path,
                                  pdf_sha: str,
                                  logger: BoundLogger,
                                  fm: FileManager,
                                  ) -> ExtractedTextResult:
    result: ExtractedTextResult | None = None
    keep_file = False
    temp_dist = fm.create_temp_file(".jsonl.part")
    try:
        temp_dist.parent.mkdir(parents=True, exist_ok=True)
        with fm.load_data(pdf_path, mode='rb') as pdf_file:
            pdf_bytes = pdf_file.read()

        pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
        with open(temp_dist, "w", encoding="utf-8", newline='') as stream:
            for i, page in enumerate(pages):
                page_text = extract_text_from_page(page)
                page_meta = {"page_num": i + 1, "language": "mr", "ocr_engine": "tesseract"}
                stream.write(
                    DocRaw(
                        id=pdf_sha + '#' + str(i + 1),
                        text=page_text,
                        meta=page_meta,
                    ).model_dump_json(ensure_ascii=False)
                )
                stream.write("\n")
        keep_file = True
        result = ExtractedTextResult(tmp_file=temp_dist, text_sha=fs.calculate_sha256(temp_dist))
    except pytesseract.TesseractError as exc:
        result = ExtractedTextResult(tmp_file=None, text_sha=None, error=str(exc))
    finally:
        if not keep_file:
            try:
                temp_dist.unlink(missing_ok=True)
            except Exception:  # noqa BLE:001
                logger.warning("failed to cleanup temp file", extra={"path": str(temp_dist)})
    return result


def run_pipeline(
    output_base_path: Path,
    database_dsn: str | None,
    manifest_path: Path,
    sync_with_fetched_items: bool,
    log: BoundLogger,
    target_status: ProcessingStatus | None = None,
    max_count: int | None = None,
    data_path: Path = Path("data"),
) -> None:
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    output_base_rel = _ensure_relative(Path(output_base_path), data_path, "output_base_path")
    output_base_abs = data_path / output_base_rel
    output_base_abs.mkdir(parents=True, exist_ok=True)

    manifest_path_abs = _ensure_absolute(Path(manifest_path), data_path)
    manifest_path_abs.parent.mkdir(parents=True, exist_ok=True)
    db_conn = get_conn(database_dsn)

    s3_cfg = load_s3_config()
    file_manager = FileManager(
        local_cfg=LocalConfig(
            base_path=data_path,
            save_local=True,
            cache_files=True,
            temp_base=data_path / "temp/extract_text",
        ),
        cloud_cfg=CloudConfig(enable=s3_cfg.enable, base_path=Path("data"), s3_config=s3_cfg, max_upload_threads=8),
    )

    manifest = ManifestStore(manifest_path_abs, ProcessedPdfManifestRecord, file_manager)
    pdf_states = PdfStateStore(db_conn)
    try:
        if sync_with_fetched_items:
            pdf_states.sync_with_fetched_items()

        task_queue: TaskQueue[PdfState] = TaskQueue(log.bind(service='TaskQueue'))
        tasks = pdf_states.filter_by_text_status(target_status)
        for task in tasks:
            normalized_pdf_path = _ensure_relative(Path(task.pdf_path), data_path, "pdf_path")
            task_queue.put(task.model_copy(update={"pdf_path": str(normalized_pdf_path)}))
            if max_count is not None and task_queue.qsize() >= max_count:
                break
        log.info(f'pipeline start working in {task_queue.qsize()} tasks, target status: {target_status}')
        result_queue: Queue[ProcessedPdfManifestRecord] = Queue()

        def extract_text_worker(worker_log: BoundLogger):
            while True:
                try:
                    task = task_queue.get(timeout=10)
                except Empty:
                    return

                try:
                    worker_result = None
                    temp_result = extract_text_from_pdf_to_temp(
                        Path(task.pdf_path), task.pdf_sha, worker_log, file_manager
                    )
                    if temp_result.tmp_file is not None and temp_result.text_sha:
                        cached_item = pdf_states.get_text_sha(temp_result.text_sha)
                        if cached_item is not None:
                            temp_result.tmp_file.unlink(missing_ok=True)
                            worker_log.debug(
                                'file skipped',
                                extra={"pdf-file": task.pdf_path,
                                       "sha256": temp_result.text_sha,
                                       "reason": "cached file"}
                            )
                            worker_result = ProcessedPdfManifestRecord(
                                pdf_path=task.pdf_path,
                                pdf_sha=task.pdf_sha,
                                status=ProcessedPdfManifestRecord.STATUS_SKIPPED,
                                text_path=cached_item.text_path,
                                text_sha=temp_result.text_sha,
                            )
                        else:
                            shard_relative = output_base_rel / file_manager.shard_relative_path(
                                temp_result.text_sha, '.jsonl'
                            )
                            file_manager.collect_result(temp_result.tmp_file, shard_relative)

                            worker_result = ProcessedPdfManifestRecord(
                                pdf_path=task.pdf_path,
                                pdf_sha=task.pdf_sha,
                                status=ProcessedPdfManifestRecord.STATUS_OK,
                                text_path=str(shard_relative),
                                text_sha=temp_result.text_sha,
                            )
                    else:
                        worker_log.warning(
                            'error extracting text',
                            extra={"pdf-file": task.pdf_path, "error": temp_result.error}
                        )
                        worker_result = ProcessedPdfManifestRecord(
                            pdf_path=task.pdf_path,
                            pdf_sha=task.pdf_sha,
                            status=ProcessedPdfManifestRecord.STATUS_ERROR,
                            text_path=None,
                            text_sha=None,
                            error=temp_result.error
                        )
                    result_queue.put(worker_result)
                except Exception as exc:
                    worker_log.exception("unexpected error during extraction", extra={"pdf-file": task.pdf_path})
                    result_queue.put(
                        ProcessedPdfManifestRecord(
                            pdf_path=task.pdf_path,
                            pdf_sha=task.pdf_sha,
                            status=ProcessedPdfManifestRecord.STATUS_ERROR,
                            text_path=None,
                            error=str(exc),
                        )
                    )
                finally:
                    task_queue.task_done()

        work_threads = [threading.Thread(target=extract_text_worker, args=(log.bind(thread_name=f'extract_worker_{ind}'),))
                        for ind in range(12)]
        for work_thread in work_threads:
            work_thread.start()

        while any(thread.is_alive() for thread in work_threads) or not result_queue.empty():
            try:
                result = result_queue.get(timeout=10)
                manifest.mark(result)
                if result.status in (ProcessedPdfManifestRecord.STATUS_OK, ProcessedPdfManifestRecord.STATUS_SKIPPED):
                    if result.text_sha and result.text_path:
                        pdf_states.update_text_status(
                            result.pdf_sha,
                            ProcessingStatus.OK,
                            result.text_sha,
                            result.text_path,
                            result.ts,
                        )
                elif result.status == ProcessedPdfManifestRecord.STATUS_ERROR:
                    pdf_states.update_text_status(
                        result.pdf_sha,
                        ProcessingStatus.ERROR,
                        result.text_sha,
                        result.text_path,
                        result.ts,
                    )
            except Empty:
                if task_queue.unfinished_tasks == 0 and result_queue.empty():
                    break
                continue

        for thread in work_threads:
            thread.join()

        task_queue.join()
        task_queue.stop_logging()

        log.info("extracting text completed, closing files")
        manifest.flush()
    finally:
        pdf_states.close()
        file_manager.close()
        log.info("files closed")


def main(argv: Iterable[str] | None = None) -> None:
    arg_parser = argparse.ArgumentParser(
        description='Extract text from pdf files'
    )
    arg_parser.add_argument('--output-base-path', required=True, help='output base path', type=Path)
    arg_parser.add_argument(
        '--database-dsn',
        '--database-path',
        '--database-url',
        dest='database_dsn',
        required=False,
        help='PostgreSQL DSN; if omitted, falls back to CCLANG_DB_DSN env var',
        type=str,
        default=None,
    )
    arg_parser.add_argument('--manifest-path', required=False, help='path to manifest', type=Path, default=None)
    arg_parser.add_argument('--sync-with-fetched-items', action='store_true', help='sync with fetched items')
    arg_parser.add_argument('--target-status', choices=["none", ProcessingStatus.OK.value, ProcessingStatus.ERROR.value],
                            default="none", help="Which text_status to process (none = not processed yet)")
    arg_parser.add_argument('--log-format', choices=["console", "json"], default="console")
    arg_parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    arg_parser.add_argument("--log-file", default=None, help="File for logs (JSONL)")
    arg_parser.add_argument("--head", required=False, default=None,
                            help="Max count of links that will be proccessed", type=int)
    arg_parser.add_argument(
        '--data-path',
        required=False,
        default=None,
        type=Path,
        help='Root directory for pipeline artifacts (defaults to ./data)',
    )

    args = arg_parser.parse_args(argv)

    manifest_path = args.manifest_path or Path("data/manifests/pl_extract_text.jsonl")
    database_dsn = args.database_dsn
    data_path = args.data_path or Path("data")

    log_file = Path(args.log_file or Path("data/logs/corpus/extract_text.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(
        fmt=args.log_format,
        level=args.log_level,
        file=str(log_file),
    )
    log = get_logger("extract_text")
    max_count = args.head if args.head is not None else None
    target_status = None if args.target_status == "none" else ProcessingStatus(args.target_status)
    try:
        return run_pipeline(
            output_base_path=args.output_base_path,
            database_dsn=database_dsn,
            manifest_path=manifest_path,
            sync_with_fetched_items=args.sync_with_fetched_items,
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
