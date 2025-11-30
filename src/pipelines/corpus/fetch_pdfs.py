"""Pipeline to download PDFs listed in a manifest, store them locally, and mirror to S3."""
import argparse
import json
import logging
import signal
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable, List

from pydantic import HttpUrl

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.ingest.fs import CloudConfig, FileManager, LocalConfig
from cclang.ingest.net import download_file_to_temp
from cclang.io.db import get_conn
from cclang.io.fetched_items_store import FetchedItemsStore
from cclang.io.manifest import ManifestStore
from cclang.io.schemas import FetchManifestRecord, SourcePDF
from cclang.models.tasks_queue import TaskQueue


def get_fetch_tasks(info_path: Path, max_count: int | None = None) -> List[HttpUrl]:
    """Load SourcePDF URLs from JSONL file, optionally limiting count."""
    tasks: List[HttpUrl] = []
    with open(info_path, 'r', encoding="utf-8") as f:
        for line in f:
            record = SourcePDF(**json.loads(line.strip()))
            tasks.append(record.url)
            if max_count is not None and len(tasks) >= max_count:
                break
    return tasks


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


def run_pipeline(
    urls_path: Path,
    output_base_path: Path,
    manifest_path: Path,
    database_path: Path,
    log: BoundLogger,
    max_count: int | None = None,
    data_path: Path = Path("data"),
) -> None:
    """Download PDFs concurrently, avoid duplicates, write manifest, mirror to S3 if enabled."""
    stop_event = threading.Event()
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    output_base_rel = _ensure_relative(Path(output_base_path), data_path, "output_base_path")
    output_base_abs = data_path / output_base_rel
    output_base_abs.mkdir(parents=True, exist_ok=True)

    manifest_path_abs = _ensure_absolute(Path(manifest_path), data_path)
    manifest_path_abs.parent.mkdir(parents=True, exist_ok=True)

    database_path_abs = _ensure_absolute(Path(database_path), data_path)
    database_path_abs.parent.mkdir(parents=True, exist_ok=True)

    db_conn = get_conn(database_path_abs)

    s3_cfg = load_s3_config()
    file_manager = FileManager(
        local_cfg=LocalConfig(
            base_path=data_path,
            save_local=True,
            cache_files=True,
            temp_base=data_path / "temp/downloads",
        ),
        cloud_cfg=CloudConfig(enable=s3_cfg.enable, base_path=Path(""), s3_config=s3_cfg, max_upload_threads=8),
    )

    fetched_items = FetchedItemsStore(db_conn)
    manifest = ManifestStore(manifest_path_abs, FetchManifestRecord)

    tasks = get_fetch_tasks(urls_path, max_count=max_count)
    tasks = list(filter(lambda task_: not fetched_items.has_url(task_), tasks))

    task_queue: TaskQueue[HttpUrl] = TaskQueue(log.bind(service='TaskQueue'))
    for task in tasks:
        task_queue.put(task)
    result_queue: Queue[FetchManifestRecord] = Queue()

    def download_worker(worker_log: logging.Logger) -> None:
        while True:
            try:
                task_url = task_queue.get(timeout=10)
            except Empty:
                if stop_event.is_set():
                    return
                continue

            try:
                if task_url is None:
                    task_queue.task_done()
                    return

                if stop_event.is_set():
                    task_queue.task_done()
                    return

                temp_result = download_file_to_temp(str(task_url), fm=file_manager)
                if temp_result.tmp_file is not None and temp_result.sha256:
                    cached_item = fetched_items.get_sha256(temp_result.sha256)
                    if cached_item is not None:
                        # already have this file on disk, reuse and drop temp
                        temp_result.tmp_file.unlink(missing_ok=True)
                        worker_log.debug(
                            "file skipped",
                            extra={"url": task_url, "sha256": temp_result.sha256, "reason": "cached file"},
                        )
                        result = FetchManifestRecord(
                            url=task_url,
                            status=FetchManifestRecord.STATUS_SKIPPED,
                            status_code=temp_result.status,
                            sha=temp_result.sha256,
                            size=temp_result.size_bytes,
                            local_path=cached_item.local_path,
                        )
                    else:
                        temp_result_path = temp_result.tmp_file
                        shard_relative = output_base_rel / file_manager.shard_relative_path(
                            temp_result.sha256, '.pdf'
                        )
                        new_result_path_abs = file_manager.resolve_local(shard_relative)
                        if new_result_path_abs.exists():
                            temp_result_path.unlink(missing_ok=True)
                        else:
                            file_manager.collect_result(temp_result_path, shard_relative)
                        worker_log.debug(
                            "file downloaded",
                            extra={"url": task_url, "sha256": temp_result.sha256, "path": str(shard_relative)},
                        )
                        result = FetchManifestRecord(
                            url=task_url,
                            status=FetchManifestRecord.STATUS_OK,
                            status_code=temp_result.status,
                            sha=temp_result.sha256,
                            size=temp_result.size_bytes,
                            local_path=str(shard_relative),
                        )
                else:
                    worker_log.warning(
                        "error downloading file",
                        extra={"url": task_url, "status": temp_result.status, "error": temp_result.error},
                    )
                    result = FetchManifestRecord(
                        url=task_url,
                        status=FetchManifestRecord.STATUS_ERROR,
                        status_code=temp_result.status,
                        sha=None,
                        size=None,
                        local_path=None,
                        error=temp_result.error
                    )
                result_queue.put(result)
            except Exception as exc:
                worker_log.exception("unexpected error during download", extra={"url": task_url})
                result_queue.put(
                    FetchManifestRecord(
                        url=task_url,
                        status=FetchManifestRecord.STATUS_ERROR,
                        status_code=None,
                        sha=None,
                        size=None,
                        local_path=None,
                        error=str(exc),
                    )
                )
            finally:
                task_queue.task_done()

    work_threads = [threading.Thread(target=download_worker, args=(log.bind(thread_name=f'download_worker_{ind}'),))
                    for ind in range(12)]
    thread_count = len(work_threads)

    def _request_stop(sig=None, frame=None):
        if stop_event.is_set():
            return
        log.warning("stop requested", extra={"signal": sig})
        stop_event.set()
        for _ in range(thread_count):
            task_queue.put(None)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, _request_stop)

    try:
        for work_thread in work_threads:
            work_thread.start()

        while any(thread.is_alive() for thread in work_threads) or not result_queue.empty():
            if stop_event.is_set() and result_queue.empty():
                break
            try:
                result = result_queue.get(timeout=2)
                manifest.mark(result)
                if result.status in (FetchManifestRecord.STATUS_OK, FetchManifestRecord.STATUS_SKIPPED):
                    if result.sha and result.local_path:
                        fetched_items.update_fetch_item(str(result.url), result.sha, result.local_path, ts=result.ts)
            except Empty:
                continue

        stop_event.set()
        for _ in range(thread_count):
            task_queue.put(None)
        for thread in work_threads:
            thread.join()
        if stop_event.is_set():
            while True:
                try:
                    task_queue.get_nowait()
                    task_queue.task_done()
                except Empty:
                    break
        task_queue.join()
    finally:
        task_queue.stop_logging()
        log.info("fetching pdfs completed, closing files")
        manifest.flush()
        fetched_items.close()
        file_manager.close()
        log.info("files closed")


def main(argv: Iterable[str] | None = None) -> None:
    arg_parser = argparse.ArgumentParser(
        description='Parse pdf files from links inside json file with schema SourcePDF'
    )
    arg_parser.add_argument('--urls-path', required=True, help='path to pdf links', type=Path)
    arg_parser.add_argument('--output-base-path', required=True, help='output base path (relative to data base)', type=Path)
    arg_parser.add_argument('--manifest-path', default=None, required=False, help='path to pipeline manifest',
                            type=Path)
    arg_parser.add_argument('--database-path', default=None, required=False, help='path to database', type=Path)
    arg_parser.add_argument('--data-path', default=None, required=False,
                            help='root directory for pipeline artifacts', type=Path)
    arg_parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            required=False, help='level of logging')
    arg_parser.add_argument('--log-format', choices=['json', 'console'], required=False, default='console',
                            help="logging format")
    arg_parser.add_argument("--log-file", default=None, help="File for logs (JSONL)")
    arg_parser.add_argument("--head", required=False, default=None,
                            help="Max count of links that will be proccessed", type=int)

    args = arg_parser.parse_args(argv)

    data_path = args.data_path or Path("data")
    manifest_path = args.manifest_path or Path("manifests/pl_fetch_pdfs.jsonl")
    database_path = args.database_path or Path("databases/pipelines.sql")

    log_file = args.log_file or Path("data/logs/corpus/fetch_pdfs.log")
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(
        fmt=args.log_format,
        level=args.log_level,
        file=str(log_file),
    )
    log = get_logger("fetch_pdfs")
    max_count = args.head if args.head is not None else None
    try:
        return run_pipeline(urls_path=args.urls_path, output_base_path=args.output_base_path,
                            manifest_path=manifest_path, database_path=database_path, log=log,
                            max_count=max_count, data_path=data_path)
    except Exception:
        log.exception("unexpected error")
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
