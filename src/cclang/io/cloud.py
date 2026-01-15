"""S3-backed object store with optional async uploads, retries, and configurable connection pooling."""
import dataclasses
import logging
import threading
import time
from pathlib import Path, PurePosixPath
from queue import Queue, Empty
from botocore.config import Config
import boto3
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from abc import ABC, abstractmethod

from cclang.common import logx
from cclang.config.s3 import S3Config

logger = logx.get_logger(__name__)
_RETRYABLE_UPLOAD_ERRORS = (
    EndpointConnectionError,
    ConnectionClosedError,
    ConnectTimeoutError,
    ReadTimeoutError,
)


class UploadCallBack(ABC):
    """Interface for upload callbacks to report success or failure."""
    @abstractmethod
    def on_upload_succes(self):
        pass

    @abstractmethod
    def on_upload_failed(self, exc_type, exc_value, traceback):
        pass


class EmptyUploadCallBack(UploadCallBack):
    """No-op callback used as a default placeholder."""
    def __init__(self):
        pass

    def on_upload_succes(self):
        pass

    def on_upload_failed(self, exc_type, exc_value, traceback):
        pass

@dataclasses.dataclass
class CloudConfig:
    """Settings for cloud storage mirroring."""
    enable: bool
    base_path: Path
    max_upload_threads: int
    s3_config: S3Config
    max_pool_connections: int | None = None

class S3Store:
    """S3/Yandex Object Storage client with optional async uploads and retry logic."""
    def __init__(
        self,
        cfg: S3Config,
        max_upload_threads: int = 8,
        upload_max_attempts: int = 3,
        upload_base_backoff: float = 0.5,
        max_pool_connections: int | None = None,
    ):
        if not cfg.enable:
            raise ValueError("S3 store is disabled")
        self.cfg = cfg
        pool_connections = max_pool_connections
        if pool_connections is None:
            pool_connections = max(32, max_upload_threads * 4)
        self.client = boto3.client(
            "s3",
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            endpoint_url="https://storage.yandexcloud.net",
            config=Config(
                max_pool_connections=pool_connections,
                connect_timeout=10,
                read_timeout=60,
            )
        )
        self.upload_max_attempts = max(upload_max_attempts, 1)
        self.upload_base_backoff = max(upload_base_backoff, 0.0)
        self.max_pool_connections = pool_connections
        self.max_upload_threads = max_upload_threads
        if max_upload_threads > 0:
            self.upload_queue: Queue[tuple[tuple[Path, Path], UploadCallBack] | None] | None = Queue()
            self.upload_threads = [
                threading.Thread(target=self._upload_worker,
                                 args=(logger.bind(worker=f'upload_worker_{idx}'),),
                                 daemon=False)
                for idx in range(max_upload_threads)
            ]
            for thread in self.upload_threads:
                thread.start()
        else:
            self.upload_queue = None
            self.upload_threads = None

    def _upload_worker(self, worker_logger: logx.BoundLogger) -> None:
        """Background worker that drains the queue and processes uploads."""
        while True:
            try:
                task_info: tuple[tuple[Path, Path], UploadCallBack] | None
                task_info = self.upload_queue.get(timeout=10)
            except Empty:
                continue

            if task_info is None:
                self.upload_queue.task_done()
                break

            task, cb = task_info
            local_path, relative_key = task
            try:
                self._upload_blocking(local_path, relative_key)
                if worker_logger.isEnabledFor(logging.DEBUG):
                    worker_logger.debug(f"uploaded {relative_key} to {local_path}")
                cb.on_upload_succes()
            except Exception as exc: # noqa BLE:001
                worker_logger.exception("S3 async upload failed",
                                        extra={"path": str(local_path), "key": str(relative_key)})
                cb.on_upload_failed(type(exc), exc, exc.__traceback__)
            finally:
                self.upload_queue.task_done()

    def _to_key(self, relative: Path) -> str:
        """Build S3 object key from a relative path and root_prefix."""
        relative_raw = PurePosixPath(relative.as_posix().lstrip("/"))
        prefix_raw = self.cfg.root_prefix.as_posix().strip("/")
        if prefix_raw and prefix_raw != ".":
            return str(PurePosixPath(prefix_raw) / relative_raw)
        return relative_raw.as_posix()

    def exists(self, relative_key: Path) -> bool:
        """Return True if the object exists in the bucket."""
        key = self._to_key(relative_key)
        try:
            self.client.head_object(Bucket=self.cfg.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def _upload_blocking(self, local_path: Path, relative_key: Path):
        """Upload synchronously with retries on transient connection errors."""
        key = self._to_key(relative_key)
        attempt = 1
        while True:
            try:
                self.client.upload_file(str(local_path), self.cfg.bucket, key)
                return
            except _RETRYABLE_UPLOAD_ERRORS as exc:
                if attempt >= self.upload_max_attempts:
                    raise
                backoff = self.upload_base_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying S3 upload after connection issue",
                    extra={
                        "key": key,
                        "attempt": attempt,
                        "max_attempts": self.upload_max_attempts,
                    },
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                if backoff:
                    time.sleep(backoff)
                attempt += 1

    def upload(self, local_path: Path, relative_key: Path, blocking: bool = True, callback: UploadCallBack = None):
        """Upload file either synchronously or enqueue for async processing; invoke callbacks."""
        if callback is None:
            callback = EmptyUploadCallBack()
        if blocking or self.upload_queue is None:
            try:
                self._upload_blocking(local_path, relative_key)
                callback.on_upload_succes()
            except Exception as exc:  # noqa: BLE001
                callback.on_upload_failed(type(exc), exc, exc.__traceback__)
                raise
        else:
            self.upload_queue.put(((local_path, relative_key), callback))

    def download(self, relative_key: Path, dest_local_path: Path):
        """Download a single object to the given local path."""
        key = self._to_key(relative_key)
        self.client.download_file(self.cfg.bucket, key, str(dest_local_path))

    def close(self):
        """Drain the upload queue and stop worker threads."""
        if self.upload_queue is None or self.upload_threads is None:
            return
        for _ in self.upload_threads:
            self.upload_queue.put(None)
        self.upload_queue.join()
        for thread in self.upload_threads:
            thread.join()
