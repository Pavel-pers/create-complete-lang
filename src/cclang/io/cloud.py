"""S3-backed object store with optional async uploads, retries, and configurable connection pooling."""
import logging
import random
import threading
import time
from pathlib import Path, PurePosixPath
from queue import Queue, Empty
from typing import Optional, Dict, Callable, Type, List

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
from dataclasses import dataclass

from cclang.io.schemas import UploadStatus, UploadManifestRecord
from cclang.core.manifest import ManifestStore
from cclang.common import logx
from cclang.config.s3 import S3Config

logger = logx.get_logger(__name__)
_RETRYABLE_CLOUD_ERRORS = (
    EndpointConnectionError,
    ConnectionClosedError,
    ConnectTimeoutError,
    ReadTimeoutError,
)


class CloudStoreError(Exception):
    """Base exception for S3Store."""


class CloudUploadError(CloudStoreError):
    """Raised when a blocking upload fails."""


class CloudDownloadError(CloudStoreError):
    """Raised when a blocking download fails."""


@dataclass
class S3Mapping:
    loc_path: Path
    cloud_key: Path


class S3JobCallback(ABC):
    """Interface for upload callbacks to report success or failure."""

    @abstractmethod
    def on_success(self, mapping: S3Mapping, extra: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_retry(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_failed(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        pass


@dataclass
class S3Job:
    mapping: S3Mapping
    callback: S3JobCallback


@dataclass
class _JobResult:
    ok: bool
    exc: Optional[Exception]
    retryable: Optional[bool]


class DefaultUploadCallback(S3JobCallback):
    def __init__(self, cb_logger: None | logx.BoundLogger = None):
        super().__init__()
        self.logger = cb_logger or logger

    def on_success(self, mapping: S3Mapping, extra: Optional[Dict] = None) -> None:
        self.logger.info(f"Upload to cloud successful: {mapping.loc_path} -> {mapping.cloud_key}")

    def on_retry(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        extra = f"#{extra['attempt']}: " if 'attempt' in extra else ''
        self.logger.warning(
            f"{extra}Retrying to upload: {mapping.loc_path} -> {mapping.cloud_key}. Exception type: {exc_type}",
            extra={"exc_value": exc_value}
        )

    def on_failed(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        self.logger.error(
            f"Error uploading file: {mapping.loc_path} -> {mapping.cloud_key}. Exception type: {exc_type}",
            extra={"exc_value": exc_value})


class DefaultDownloadCallback(S3JobCallback):
    def __init__(self, cb_logger: None | logx.BoundLogger = None):
        super().__init__()
        self.logger = cb_logger or logger

    def on_success(self, mapping: S3Mapping, extra: Optional[Dict] = None) -> None:
        self.logger.info(f"Download from cloud successful: {mapping.loc_path} <- {mapping.cloud_key}")

    def on_retry(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        extra = f"#{extra['attempt']}: " if 'attempt' in extra else ''
        self.logger.warning(
            f"{extra}Retrying to download: {mapping.loc_path} <- {mapping.cloud_key}. Exception type: {exc_type}",
            extra={"exc_value": exc_value}
        )

    def on_failed(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        self.logger.error(f"Error download file: {mapping.loc_path} <- {mapping.cloud_key}. Exception type: {exc_type}",
                          extra={"exc_value": exc_value})


class S3Store:
    """S3/Yandex Object Storage client with optional async uploads and retry logic."""

    def __init__(
            self,
            cfg: S3Config,
            transfer_manifest: ManifestStore[UploadManifestRecord],
            max_upload_threads: int = 8,
            max_download_threads: int = 0,
            cloud_max_attempts: int = 4,
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

        self.transfer_manifest = transfer_manifest
        self.cloud_max_attempts = max(cloud_max_attempts, 1)
        self.cloud_base_backoff = max(upload_base_backoff, 0.0)
        self.max_pool_connections = pool_connections
        self.max_upload_threads = max_upload_threads

        self.upload_queue: Optional[Queue[S3Job]] = None
        self.download_queue: Optional[Queue[S3Job]] = None
        self.upload_threads: List[threading.Thread] = []
        self.download_threads: List[threading.Thread] = []

        if max_upload_threads > 0:
            self.upload_queue = Queue()
            self.upload_threads = [
                threading.Thread(target=self._upload_worker,
                                 args=(logger.bind(worker=f'upload_worker_{idx}'),),
                                 daemon=False)
                for idx in range(max_upload_threads)
            ]
            for thread in self.upload_threads:
                thread.start()

        if max_download_threads > 0:
            raise NotImplementedError("Multithread download are not yet supported")

    def _update_job_status(self, job: S3Job, status: UploadStatus):
        try:
            self.transfer_manifest.mark(
                UploadManifestRecord(
                    local_path=job.mapping.loc_path,
                    cloud_key=job.mapping.cloud_key,
                    status=status,
                ))
        except Exception as exc:  # noqa :BLE001
            logger.exception("Failed to write into transfer manifest",
                             extra={"status": str(status),
                                    "local_path": str(job.mapping.loc_path),
                                    "cloud_key": str(job.mapping.cloud_key)})

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

    def _run_job_with_retries(
            self,
            job: S3Job,
            *,
            action: Callable[[], None],
            retryable_errors: tuple[Type[BaseException], ...],
            max_attempts: int,
            base_backoff: float,
    ) -> _JobResult:
        """
        Run an operation with exponential backoff + jitter for retryable errors.

        Notes:
        - This helper is responsible for calling job callbacks (success/retry/failure).
        - It does not raise on failure; it returns True/False.
        """
        attempt = 1
        max_attempts = max(1, max_attempts)
        base_backoff = max(0.0, base_backoff)

        while True:
            try:
                action()
                job.callback.on_success(
                    job.mapping
                )
                return _JobResult(ok=True)

            except retryable_errors as exc:
                # Reached max attempts or retryable error
                if attempt >= max_attempts:
                    job.callback.on_failed(
                        job.mapping,
                        type(exc),
                        exc,
                        exc.__traceback__,
                        extra={"attempt": attempt},
                    )
                    return _JobResult(ok=False, exc=exc, retryable=True)

                # Retry job
                job.callback.on_retry(
                    job.mapping,
                    type(exc),
                    exc,
                    exc.__traceback__,
                    extra={"attempt": attempt},
                )

                # Exponential backoff
                backoff = base_backoff * (2 ** (attempt - 1))
                backoff *= random.uniform(0.8, 1.2)
                if backoff:
                    time.sleep(backoff)

                attempt += 1

            except Exception as exc:  # noqa: BLE001
                # Non-retryable failure
                job.callback.on_failed(
                    job.mapping,
                    type(exc),
                    exc,
                    exc.__traceback__,
                )

                return _JobResult(ok=False, exc=exc, retryable=False)

    def _do_upload_job(self, upload_job: S3Job) -> _JobResult:
        """
        Try to upload a single job. If not successful, retry with backoff, up to max_attempts. Return True on success.
        Call callbacks its own
        :param upload_job:
        :return: True if upload succeeded, False otherwise.
        """
        cloud_key = self._to_key(upload_job.mapping.cloud_key)
        local_path = upload_job.mapping.loc_path

        def upload_action() -> None:
            self.client.upload_file(str(local_path), self.cfg.bucket, cloud_key)

        return self._run_job_with_retries(upload_job,
                                          action=upload_action,
                                          retryable_errors=_RETRYABLE_CLOUD_ERRORS,
                                          max_attempts=self.cloud_max_attempts,
                                          base_backoff=self.cloud_base_backoff, )

    def _upload_worker(self, worker_logger: logx.BoundLogger) -> None:
        """Background worker that drains the queue and processes uploads."""
        while True:
            try:
                job_item = self.upload_queue.get(timeout=10)
            except Empty:
                continue

            if job_item is None:
                self.upload_queue.task_done()
                break

            self._update_job_status(job_item, UploadStatus.STARTED)

            result = self._do_upload_job(job_item)

            if result.ok:
                self._update_job_status(job_item, UploadStatus.SUCCEDED)
            else:
                self._update_job_status(job_item, UploadStatus.FAILED)

            self.upload_queue.task_done()

    def _do_download_job(self, download_job: S3Job) -> _JobResult:
        """
        Executes a download job from the cloud to a local path.

        :param download_job: The S3Job object containing the details of the download job.
        :type download_job: S3Job
        :return: A boolean indicating the success or failure of the download operation.
        :rtype: bool
        """
        cloud_key = self._to_key(download_job.mapping.cloud_key)
        local_path = download_job.mapping.loc_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = local_path.with_name(local_path.name + ".part")

        def download_action() -> None:
            self.client.download_file(self.cfg.bucket, cloud_key, str(part_path))
            part_path.replace(local_path)

        result = self._run_job_with_retries(download_job,
                                            action=download_action,
                                            retryable_errors=_RETRYABLE_CLOUD_ERRORS,
                                            max_attempts=self.cloud_max_attempts,
                                            base_backoff=self.cloud_base_backoff, )

        if not result.ok:
            if part_path.exists():
                part_path.unlink()

        return result

    def upload(self, local_path: Path, relative_key: Path, blocking: bool = True,
               callback: S3JobCallback = None) -> None:
        """
        Upload a file to an S3 storage either synchronously or asynchronously, depending on
        the ``blocking`` parameter. Invoke the provided callback or a default callback upon
        completion of the upload operation.
        * If blocking parameter is True, the upload will block until completion, and raise exception in case error
        * If non-blocking call, inform about cloud error only in logs, and doesn't guarantee that upload will succeed.

        :param local_path: Path to the local file to be uploaded
        :type local_path: Path
        :param relative_key: The key or relative path to store the file in S3
        :type relative_key: Path
        :param blocking: Whether the upload should be processed synchronously or enqueued for
            asynchronous upload (default is True). Exception in case cloud error raises only on blocking is True,
            otherwise we don't guarantee that upload will succeed after call.
        :type blocking: bool, optional
        :param callback: Optional callback to execute upon completion of the upload job
        :type callback: S3JobCallback, optional
        :return: None
        """

        if not local_path.exists():
            raise FileNotFoundError(f"Local file {local_path} does not exist")

        if callback is None:
            callback = DefaultUploadCallback()

        upload_job = S3Job(S3Mapping(loc_path=local_path, cloud_key=relative_key), callback)

        if blocking:
            self._update_job_status(upload_job, UploadStatus.STARTED)

            result = self._do_upload_job(upload_job)
            if result.ok:
                self._update_job_status(upload_job, UploadStatus.SUCCEDED)
                return

            err = str(result.exc)
            self._update_job_status(upload_job, UploadStatus.FAILED)
            raise CloudUploadError(f"Blocking upload failed: {local_path} -> {relative_key}") from result.exc
        else:
            if self.upload_queue is None:
                raise RuntimeError("Async upload not enabled")

            self._update_job_status(upload_job, UploadStatus.QUEUED)
            self.upload_queue.put(upload_job)

    def download(self, relative_key: Path, local_path: Path, blocking: bool = True,
                 callback: S3JobCallback = None) -> None:
        """
        Downloads a file from an S3 bucket to a local path. The method supports blocking
        synchronous download by default and provides an option to specify a callback
        for handling the result of the operation. Currently, asynchronous download
        functionality is not implemented.

        :param relative_key: The key in the S3 bucket corresponding to the file
            that needs to be downloaded.
        :type relative_key: Path
        :param local_path: The local file path where the downloaded file
            will be saved.
        :type local_path: Path
        :param blocking: A flag indicating whether the operation should block
            the current thread until the download completes. Defaults to True.
        :type blocking: bool
        :param callback: An optional callback to handle the result of the
            download operation. If not specified, a default callback is used.
        :type callback: S3JobCallback or None
        :return: None
        """
        if callback is None:
            callback = DefaultDownloadCallback()

        upload_job = S3Job(mapping=S3Mapping(loc_path=local_path, cloud_key=relative_key),
                           callback=callback)

        if blocking:
            self._update_job_status(upload_job, UploadStatus.STARTED)
            result = self._do_download_job(upload_job)
            if result.ok:
                self._update_job_status(upload_job, UploadStatus.SUCCEDED)
                return
            err = str(result.exc)
            self._update_job_status(upload_job, UploadStatus.FAILED)
            raise CloudDownloadError(f"Blocking download failed: {local_path} <- {relative_key}") from result.exc
        else:
            raise NotImplementedError('Async download is not yet supported')

    def close(self):
        """Drain the upload queue and stop worker threads."""
        if self.upload_queue is None or self.upload_threads is None:
            return
        for _ in self.upload_threads:
            self.upload_queue.put(None)
        self.upload_queue.join()
        for thread in self.upload_threads:
            thread.join()
