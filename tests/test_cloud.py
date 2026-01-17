import os
import time
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch, ANY
import threading

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from cclang.io.cloud import (
    S3Store,
    S3Config,
    S3JobCallback,
    UploadStatus,
    UploadManifestRecord,
    S3Job,
    S3Mapping,
    CloudUploadError,
    CloudDownloadError,
    _RETRYABLE_CLOUD_ERRORS,
)
from cclang.core.manifest import ManifestStore


@pytest.fixture
def s3_config():
    return S3Config(
        enable=True,
        bucket="test-bucket",
        access_key="test-key",
        secret_key="test-secret",
        region="ru-central1",
        root_prefix=Path("prefix"),
    )


@pytest.fixture
def mock_manifest_store():
    manifest = MagicMock(spec=ManifestStore)
    manifest.manifest_rel_path = Path("transfer_manifest.jsonl")
    return manifest


@pytest.fixture
def s3_store(s3_config, mock_manifest_store):
    with patch("cloud.boto3.client") as mock_client:
        mock_client.return_value = MagicMock()
        store = S3Store(
            cfg=s3_config,
            transfer_manifest=mock_manifest_store,
            max_upload_threads=2,
            cloud_max_attempts=3,
            cloud_base_backoff=0.1,
        )
        return store


class TestCallback(S3JobCallback):
    def __init__(self):
        self.success_calls = 0
        self.retry_calls = 0
        self.failed_calls = 0

    def on_success(self, mapping, extra=None):
        self.success_calls += 1

    def on_retry(self, mapping, exc_type, exc_value, traceback, extra=None):
        self.retry_calls += 1

    def on_failed(self, mapping, exc_type, exc_value, traceback, extra=None):
        self.failed_calls += 1


def test_initialization_validates_config():
    invalid_config = S3Config(enable=False, bucket="", access_key="", secret_key="", region="")

    with pytest.raises(ValueError, match="S3 store is disabled"):
        S3Store(cfg=invalid_config, transfer_manifest=MagicMock())


def test_endpoint_url_no_spaces(s3_config, mock_manifest_store):
    """Тест, что endpoint_url не содержит лишних пробелов"""
    with patch("cloud.boto3.client") as mock_client:
        S3Store(
            cfg=s3_config,
            transfer_manifest=mock_manifest_store,
            max_upload_threads=0
        )
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["endpoint_url"] == "https://storage.yandexcloud.net"


def test_exists_handles_404(s3_store):
    s3_store.client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404"}}, "head_object"
    )

    assert s3_store.exists(Path("non-existent.txt")) is False


def test_exists_propagates_other_errors(s3_store):
    s3_store.client.head_object.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Access denied"}}, "head_object"
    )

    with pytest.raises(ClientError):
        s3_store.exists(Path("forbidden.txt"))


def test_to_key_handles_special_cases(s3_store):
    # Проверка различных сценариев формирования ключей
    test_cases = [
        (Path("file.txt"), "prefix/file.txt"),
        (Path("/file.txt"), "prefix/file.txt"),
        (Path("subdir/file.txt"), "prefix/subdir/file.txt"),
        (Path("/subdir/file.txt"), "prefix/subdir/file.txt"),
    ]

    for input_path, expected_key in test_cases:
        assert s3_store._to_key(input_path) == expected_key

    # Случай с пустым префиксом
    s3_store._cfg.root_prefix = Path("")
    assert s3_store._to_key(Path("file.txt")) == "file.txt"


@pytest.mark.parametrize("error_type", _RETRYABLE_CLOUD_ERRORS)
def test_upload_with_retries(s3_store, error_type):
    callback = TestCallback()
    local_file = Path("/tmp/test.txt")
    local_file.write_text("test content")

    # Первая две попытки падают с ошибкой, третья успешна
    s3_store.client.upload_file.side_effect = [
        error_type(endpoint_url="", error="", args=()),
        error_type(endpoint_url="", error="", args=()),
        None  # Успешная загрузка
    ]

    s3_store._do_upload_job(S3Job(
        mapping=S3Mapping(loc_path=local_file, cloud_key=Path("remote.txt")),
        callback=callback
    ))

    assert callback.retry_calls == 2
    assert callback.success_calls == 1
    assert s3_store.client.upload_file.call_count == 3


def test_upload_fails_after_max_attempts(s3_store):
    callback = TestCallback()
    local_file = Path("/tmp/test.txt")
    local_file.write_text("test content")

    # Все попытки падают
    s3_store.client.upload_file.side_effect = EndpointConnectionError(
        endpoint_url="", error="", args=()
    )

    result = s3_store._do_upload_job(S3Job(
        mapping=S3Mapping(loc_path=local_file, cloud_key=Path("remote.txt")),
        callback=callback
    ))

    assert callback.retry_calls == 2  # max_attempts-1 попыток с повтором
    assert callback.failed_calls == 1
    assert not result.ok
    assert result.retryable is True


def test_blocking_upload_raises_exception_on_failure(s3_store):
    local_file = Path("/tmp/test.txt")
    local_file.write_text("test content")

    s3_store.client.upload_file.side_effect = ReadTimeoutError(
        endpoint_url="", error="", args=()
    )

    with pytest.raises(CloudUploadError):
        s3_store.upload(local_file, Path("remote.txt"), blocking=True)


def test_non_blocking_upload_queues_job(s3_store):
    # Настраиваем очередь для асинхронной загрузки
    s3_store._upload_queue = Queue()
    local_file = Path("/tmp/test.txt")
    local_file.write_text("test content")

    # Мокаем метод _do_upload_job для проверки выполнения в фоне
    with patch.object(s3_store, '_do_upload_job') as mock_do_upload:
        mock_do_upload.return_value = MagicMock(ok=True)

        s3_store.upload(local_file, Path("remote.txt"), blocking=False)

        # Ждем, пока задача попадет в очередь
        job = s3_store._upload_queue.get(timeout=1)
        s3_store._upload_queue.task_done()

        assert job.mapping.loc_path == local_file
        assert job.mapping.cloud_key == Path("remote.txt")

        # Проверяем, что фоновый процесс попытается выполнить задачу
        mock_do_upload.assert_called_once()


def test_download_handles_file_operations(s3_store, tmp_path):
    # Создаем временный файл для имитации скачивания
    temp_dir = tmp_path / "download"
    temp_dir.mkdir()
    local_path = temp_dir / "downloaded.txt"
    part_path = local_path.with_name(local_path.name + ".part")

    callback = TestCallback()

    # Имитируем успешное скачивание
    def mock_download_file(bucket, key, filename):
        with open(filename, 'w') as f:
            f.write("downloaded content")

    s3_store.client.download_file.side_effect = mock_download_file

    result = s3_store._do_download_job(S3Job(
        mapping=S3Mapping(loc_path=local_path, cloud_key=Path("remote.txt")),
        callback=callback
    ))

    assert result.ok is True
    assert callback.success_calls == 1
    assert local_path.exists()
    assert local_path.read_text() == "downloaded content"
    assert not part_path.exists()  # .part файл должен быть удален


def test_download_cleans_up_on_failure(s3_store, tmp_path):
    temp_dir = tmp_path / "download"
    temp_dir.mkdir()
    local_path = temp_dir / "failed.txt"
    part_path = local_path.with_name(local_path.name + ".part")

    # Имитируем ошибку при скачивании
    s3_store.client.download_file.side_effect = ReadTimeoutError(
        endpoint_url="", error="", args=()
    )

    callback = TestCallback()
    result = s3_store._do_download_job(S3Job(
        mapping=S3Mapping(loc_path=local_path, cloud_key=Path("remote.txt")),
        callback=callback
    ))

    assert result.ok is False
    assert callback.failed_calls == 1
    assert not local_path.exists()
    assert not part_path.exists()  # .part файл должен быть удален даже при ошибке


def test_close_handles_worker_shutdown(s3_store):
    # Создаем рабочие потоки
    s3_store._upload_threads = [
        threading.Thread(target=lambda: time.sleep(0.1), daemon=True)
        for _ in range(2)
    ]

    for thread in s3_store._upload_threads:
        thread.start()

    # Имитируем очередь
    s3_store._upload_queue = MagicMock(spec=Queue)
    s3_store._upload_queue.join = MagicMock()

    s3_store.close()

    # Проверяем, что join был вызван
    s3_store._upload_queue.join.assert_called_once()

    # Проверяем, что потоки завершились
    for thread in s3_store._upload_threads:
        assert not thread.is_alive()


def test_update_job_status_handles_manifest_errors(s3_store, caplog):
    job = S3Job(
        mapping=S3Mapping(loc_path=Path("/tmp/file.txt"), cloud_key=Path("remote.txt")),
        callback=MagicMock()
    )

    # Имитируем ошибку при записи в манифест
    s3_store._transfer_manifest.mark.side_effect = Exception("Manifest write failed")

    s3_store._update_job_status(job, UploadStatus.FAILED)

    # Проверяем, что ошибка была залогирована
    assert "Failed to write into transfer manifest" in caplog.text


def test_async_download_not_implemented(s3_store, tmp_path):
    local_path = tmp_path / "download.txt"

    with pytest.raises(NotImplementedError, match="Async download is not yet supported"):
        s3_store.download(Path("remote.txt"), local_path, blocking=False)