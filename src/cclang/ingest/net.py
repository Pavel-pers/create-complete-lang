import hashlib
import requests
from requests.exceptions import RequestException
import shutil
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

from cclang.common import logx
from cclang.ingest import fs
from cclang.io.schemas import FetchManifestRecord
from cclang.io.manifest import ManifestStore

net_logger = logx.get_logger("cclang.net")


@dataclass
class DownloadResult:
    status: int
    size_bytes: int
    tmp_file: Optional[Path]
    sha256: Optional[str]
    error: Optional[str] = None


def download_file_to_temp(url, dowload_folder: Path = "data/temp/downloads") -> DownloadResult:
    with net_logger.bind(url=url) as log:
        temp_dist = fs.create_temp_file(dowload_folder, ".part")
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                if not (200 <= response.status_code <= 299):
                    log.debug(f"Download {url} failed with status {response.status_code}")
                    return DownloadResult(status=response.status_code, size_bytes=0, tmp_file=None, sha256=None,
                                          error=f"unknown status code http_{response.status_code}")

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

                log.debug('succesfull download file')
                return DownloadResult(status=200, size_bytes=size, tmp_file=temp_dist, sha256=cur_hash.hexdigest())
        except RequestException as exc:
            log.debug(f"network request failed, exception type: {type(exc)}")
            return DownloadResult(status=200, size_bytes=0, tmp_file=None, sha256=None, error=str(exc))
