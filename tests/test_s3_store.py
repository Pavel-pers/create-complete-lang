from pathlib import Path
from typing import Optional, Dict

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from cclang.config.s3 import S3Config
from cclang.io.cloud import S3Store, S3JobCallback, S3Mapping


class _FlakyClient:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0
        self.last_upload = None

    def upload_file(self, filename, bucket, key):
        self.calls += 1
        if self.calls <= self.failures:
            raise EndpointConnectionError(endpoint_url="http://example.com")
        self.last_upload = (filename, bucket, key)


class _Cb(S3JobCallback):
    def __init__(self):
        self.ok = 0
        self.failed = 0

    def on_success(self, mapping: S3Mapping, extra: Optional[Dict] = None) -> None:
        self.ok += 1

    def on_failed(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        self.failed += 1

    def on_retry(self, mapping: S3Mapping, exc_type, exc_value, traceback, extra: Optional[Dict] = None) -> None:
        pass


class _FakeManifest:
    """Minimal mock for ManifestStore used as transfer_manifest."""
    def __init__(self):
        self.records = []

    def mark(self, record):
        self.records.append(record)


def _make_cfg() -> S3Config:
    return S3Config(
        enable=True,
        bucket="bucket",
        root_prefix=Path("root"),
        region=None,
        access_key="ak",
        secret_key="sk",
    )


def test_upload_retries_on_connection_error(monkeypatch, tmp_path: Path):
    upload_path = tmp_path / "file.bin"
    upload_path.write_bytes(b"data")

    client = _FlakyClient(failures=2)
    monkeypatch.setattr("cclang.io.cloud.boto3.client", lambda *args, **kwargs: client)

    store = S3Store(
        _make_cfg(),
        transfer_manifest=_FakeManifest(),
        max_upload_threads=0,
        cloud_max_attempts=3,
        cloud_base_backoff=0,
    )

    cb = _Cb()

    store.upload(upload_path, Path("dest.bin"), blocking=True, callback=cb)

    assert client.calls == 3
    assert client.last_upload == (str(upload_path), "bucket", "root/dest.bin")
    assert cb.ok == 1
    assert cb.failed == 0


def test_upload_does_not_retry_on_client_error(monkeypatch, tmp_path: Path):
    upload_path = tmp_path / "file.bin"
    upload_path.write_bytes(b"data")

    class _FailingClient:
        def __init__(self):
            self.calls = 0

        def upload_file(self, filename, bucket, key):
            self.calls += 1
            raise ClientError(
                error_response={"Error": {"Code": "403", "Message": "denied"}},
                operation_name="UploadFile",
            )

    client = _FailingClient()
    monkeypatch.setattr("cclang.io.cloud.boto3.client", lambda *args, **kwargs: client)

    store = S3Store(
        _make_cfg(),
        transfer_manifest=_FakeManifest(),
        max_upload_threads=0,
        cloud_max_attempts=5,
        cloud_base_backoff=0,
    )

    cb = _Cb()

    with pytest.raises(Exception):
        store.upload(upload_path, Path("dest.bin"), blocking=True, callback=cb)

    assert client.calls == 1
    assert cb.ok == 0
    assert cb.failed == 1


def test_async_upload_invokes_callback(monkeypatch, tmp_path: Path):
    upload_path = tmp_path / "file.bin"
    upload_path.write_bytes(b"data")

    client = _FlakyClient(failures=0)
    monkeypatch.setattr("cclang.io.cloud.boto3.client", lambda *args, **kwargs: client)

    store = S3Store(
        _make_cfg(),
        transfer_manifest=_FakeManifest(),
        max_upload_threads=1,
        cloud_max_attempts=1,
        cloud_base_backoff=0,
    )

    cb = _Cb()
    store.upload(upload_path, Path("dest.bin"), blocking=False, callback=cb)
    store.close()

    assert client.calls == 1
    assert cb.ok == 1
