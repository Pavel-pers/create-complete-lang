import hashlib
from pathlib import Path

import pytest
from requests.exceptions import Timeout

from cclang.ingest.net import DownloadResult, download_file_to_temp


class _FakeResponse:
    def __init__(self, status_code: int, chunks: list[bytes]):
        self.status_code = status_code
        self._chunks = chunks

    def iter_content(self, chunk_size: int = 0):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_success_writes_file_and_sha(monkeypatch, tmp_path: Path):
    data = b"hello world"
    expected_sha = hashlib.sha256(data).hexdigest()
    download_dir = tmp_path / "downloads"

    def fake_get(url, stream=True, timeout=0):
        return _FakeResponse(status_code=200, chunks=[data])

    monkeypatch.setattr("cclang.ingest.net.requests.get", fake_get)

    result = download_file_to_temp("http://example.com/file.pdf", download_folder=download_dir)

    assert result.status == 200
    assert result.sha256 == expected_sha
    assert result.tmp_file is not None
    assert result.tmp_file.exists()
    assert result.tmp_file.read_bytes() == data


def test_download_non_2xx_cleans_temp(monkeypatch, tmp_path: Path):
    download_dir = tmp_path / "downloads"

    def fake_get(url, stream=True, timeout=0):
        return _FakeResponse(status_code=404, chunks=[b"oops"])

    monkeypatch.setattr("cclang.ingest.net.requests.get", fake_get)

    result = download_file_to_temp("http://example.com/missing.pdf", download_folder=download_dir)

    assert result.status == 404
    assert result.tmp_file is None
    assert not any(download_dir.glob("*.part"))


def test_download_request_exception_cleans_temp(monkeypatch, tmp_path: Path):
    download_dir = tmp_path / "downloads"

    def fake_get(url, stream=True, timeout=0):
        raise Timeout("boom")

    monkeypatch.setattr("cclang.ingest.net.requests.get", fake_get)

    result = download_file_to_temp("http://example.com/timeout.pdf", download_folder=download_dir)

    assert result.status == 0
    assert result.tmp_file is None
    assert "boom" in (result.error or "")
    assert not any(download_dir.glob("*.part"))
