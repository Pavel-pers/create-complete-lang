"""Network helpers for downloading files to temp storage with hashing and cleanup."""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import RequestException

from cclang.common import logx
from cclang.ingest import fs

net_logger = logx.get_logger("cclang.net")


@dataclass
class DownloadResult:
    """Container for download metadata and temporary file reference."""
    status: int
    size_bytes: int
    tmp_file: Optional[Path]
    sha256: Optional[str]
    error: Optional[str] = None


def download_file_to_temp(
    url: str,
    fm: fs.FileManager | None = None,
    download_folder: Path | str = Path("data/temp/downloads"),
) -> DownloadResult:
    """
    Download file to a temp location and return metadata. Cleans up temp file on errors.
    If `fm` is provided, temp files are created via FileManager.
    """
    log = net_logger.bind(url=url)
    temp_dist = fm.create_temp_file(".part") if fm is not None else fs.create_temp_file(download_folder, ".part")

    result: DownloadResult | None = None
    keep_file = False
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            if not 200 <= response.status_code <= 299:
                log.debug("download failed: unexpected status", extra={"status": response.status_code})
                result = DownloadResult(
                    status=response.status_code,
                    size_bytes=0,
                    tmp_file=None,
                    sha256=None,
                    error=f"http_{response.status_code}",
                )
                return result

            cur_hash = hashlib.sha256()
            size = 0
            temp_dist.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_dist, "wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    cur_hash.update(chunk)
                    size += len(chunk)

            log.debug("successful download", extra={"bytes": size})
            keep_file = True
            result = DownloadResult(status=200, size_bytes=size, tmp_file=temp_dist, sha256=cur_hash.hexdigest())
            return result
    except RequestException as exc:
        log.debug("network request failed", extra={"error": str(exc), "exc_type": type(exc).__name__})
        status_code = exc.response.status_code if getattr(exc, "response", None) is not None else 0
        result = DownloadResult(status=status_code, size_bytes=0, tmp_file=None, sha256=None, error=str(exc))
        return result
    finally:
        if not keep_file:
            try:
                temp_dist.unlink(missing_ok=True)
            except Exception: # noqa BLE:001
                log.warning("failed to cleanup temp file", extra={"path": str(temp_dist)})
