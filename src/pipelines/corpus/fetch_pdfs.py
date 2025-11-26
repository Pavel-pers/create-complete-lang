import argparse
import sys
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import List, Iterable
import logging
import json
from pydantic import HttpUrl

from cclang.io.db import get_conn
from cclang.io.manifest import ManifestStore
from cclang.io.fetched_items_store import FetchedItemsStore
from cclang.io.schemas import FetchManifestRecord, SourcePDF
from cclang.ingest.net import download_file_to_temp
from cclang.ingest.fs import get_shard_path, atomic_move
from cclang.common.logx import setup_logging, get_logger, BoundLogger
from cclang.models.tasks_queue import TaskQueue


def get_fetch_tasks(info_path: Path, max_count: int = None) -> List[HttpUrl]:
    tasks: List[HttpUrl] = []
    with open(info_path, 'r', encoding="utf-8") as f:
        for line in f:
            record = SourcePDF(**json.loads(line.strip()))
            tasks.append(HttpUrl(record.url))
            if max_count is not None and len(tasks) >= max_count:
                break
    return tasks


def run_pipeline(urls_path: Path, output_base_path: Path, manifest_path: Path, database_path: Path,
                 log: BoundLogger) -> None:
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    db_conn = get_conn(database_path)
    tasks = get_fetch_tasks(urls_path)

    fetched_items = FetchedItemsStore(db_conn)
    manifest = ManifestStore(manifest_path, FetchManifestRecord)

    task_queue: TaskQueue[HttpUrl] = TaskQueue(log.bind(service='TaskQueue'))
    for task in tasks:
        task_queue.put(task)
    result_queue: Queue[FetchManifestRecord] = Queue()

    def download_worker(worker_log: logging.Logger) -> None:
        while not task_queue.empty():
            try:
                task_url = task_queue.get(timeout=10)
                temp_result = download_file_to_temp(task_url)
                if temp_result.tmp_file is not None:
                    if fetched_items.has_sha256(temp_result.sha256):
                        cached_file_path = fetched_items.get_sha256(temp_result.sha256).local_path
                        worker_log.debug(f"file skipped",
                                         extra={"url": task_url, "sha256": temp_result.sha256, "reason": "cached file"})
                        result = FetchManifestRecord(
                            url=HttpUrl(task_url),
                            status=FetchManifestRecord.STATUS_SKIPPED,
                            status_code=temp_result.status,
                            sha=temp_result.sha256,
                            size=temp_result.size_bytes,
                            local_path=cached_file_path,
                        )
                        result_queue.put(result)
                    else:
                        temp_result_path = temp_result.tmp_file
                        new_result_path = get_shard_path(output_base_path, temp_result.sha256, '.pdf')
                        atomic_move(temp_result_path, new_result_path)
                        worker_log.debug(f"file downloaded",
                                         extra={"url": task_url, "sha256": temp_result.sha256, "path": new_result_path})
                        result = FetchManifestRecord(
                            url=HttpUrl(task_url),
                            status=FetchManifestRecord.Status.DOWNLOADED,
                            status_code=temp_result.status,
                            sha=temp_result.sha256,
                            size=temp_result.size_bytes,
                            local_path=str(new_result_path)
                        )
                        result_queue.put(result)
                else:
                    worker_log.warning(f"error downloading file",
                                       extra={"url": task_url, "status": temp_result.status,
                                              "error": temp_result.error})
                    result = FetchManifestRecord(
                        url=HttpUrl(task_url),
                        status=FetchManifestRecord.Status.ERROR,
                        status_code=temp_result.status,
                        sha=None,
                        size=None,
                        local_path=None,
                        error=temp_result.error
                    )
                    result_queue.put(result)

                task_queue.task_done()
            except Empty:
                continue

    work_threads = [threading.Thread(target=download_worker, args=(log.bind(thread_name=f'download_worker_{ind}'),))
                    for ind in range(8)]
    for work_thread in work_threads:
        work_thread.start()

    while any(thread.is_alive() for thread in work_threads) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=10)
            manifest.mark(result)
        except Empty:
            continue

    task_queue.join()
    task_queue.stop_logging()
    manifest.flush()

    db_conn.close()
    fetched_items.close()


def main(argv: Iterable[str] | None) -> None:
    arg_parser = argparse.ArgumentParser(
        description='Parse pdf files from links inside json file with schema SourcePDF'
    )
    arg_parser.add_argument('--urls-path', required=True, help='path to pdf links', type=Path)
    arg_parser.add_argument('--output-base-path', required=True, help='output base path', type=Path)
    arg_parser.add_argument('--manifest-path', default=None, required=False, help='path to pipeline manifest', type=Path)
    arg_parser.add_argument('--database-path', default=None, required=False, help='path to database', type=Path)
    arg_parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            required=False, help='level of logging')
    arg_parser.add_argument('--log-format', choices=['json', 'console'], required=False, default='console',
                            help="logging format")
    arg_parser.add_argument("--log-file", default=None, help="Файл для записи логов (JSONL)")
    args = arg_parser.parse_args(argv)

    setup_logging(
        fmt = args.log_format,
        level = args.log_level,
        file = arg_parser or 'data/logs/corpus/fetch_pdfs.log',
    )
    log = get_logger("fetch_pdfs")

    try:
        return run_pipeline(urls_path = args.urls_path, output_base_path = args.output_base_path, manifest_path=args.manifest_path, database_path=args.database_path, log=log)
    except Exception:
        log.exception("unexpected error")
        raise

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
