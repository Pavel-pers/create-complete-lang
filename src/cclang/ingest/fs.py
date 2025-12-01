"""Local + cloud file management utilities: sharded paths, temp files, and S3 uploads/downloads."""
from __future__ import annotations
import dataclasses
import hashlib
import os
import uuid
from os import unlink
from pathlib import Path

from cclang.config.s3 import S3Config
from cclang.io.cloud import S3Store, UploadCallBack


def _create_temp_file(base_path: Path | str, suff: str) -> Path:
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    name = str(uuid.uuid4()) + suff
    return base_path / name


def create_temp_file(base_path: Path | str, suff: str = "") -> Path:
    """
    Public helper kept for backward compatibility.
    """
    return _create_temp_file(base_path, suff)


def _get_shard_relative(file_name: str, file_suffix: str) -> Path:
    shard_folder = file_name[:2].zfill(2)
    return Path(shard_folder) / (file_name + file_suffix)


def get_shard_path(base: Path | str, file_name: str, file_suffix: str) -> Path:
    """
    Public helper kept for backward compatibility.
    """
    return Path(base) / _get_shard_relative(file_name, file_suffix)


def calculate_sha256(path: Path | str) -> str:
    path = Path(path)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_move(temp_file_path: Path | str, dest_path: Path | str) -> None:
    """
    Atomic move temp downloaded file to shard folder.
    """
    dest_path = Path(dest_path)
    temp_file_path = Path(temp_file_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_file_path, dest_path)


@dataclasses.dataclass
class LocalConfig:
    """Settings for local storage: base path, caching, and temp files."""
    base_path: Path
    save_local: bool
    cache_files: bool
    temp_base: Path


@dataclasses.dataclass
class CloudConfig:
    """Settings for cloud storage mirroring."""
    enable: bool
    base_path: Path
    max_upload_threads: int
    s3_config: S3Config
    max_pool_connections: int | None = None


class FileManager:
    """Handles sharded local paths and optional S3 mirroring."""
    def __init__(self, local_cfg: LocalConfig, cloud_cfg: CloudConfig):
        self.local_cfg = local_cfg
        self.cloud_cfg = cloud_cfg
        self.cloud: S3Store | None = None
        if self.cloud_cfg.enable:
            self.cloud = S3Store(
                self.cloud_cfg.s3_config,
                self.cloud_cfg.max_upload_threads,
                max_pool_connections=self.cloud_cfg.max_pool_connections,
            )

    def create_temp_file(self, suffix: str = "") -> Path:
        return _create_temp_file(self.local_cfg.temp_base, suffix)

    def shard_relative_path(self, file_name: str, suffix: str = "") -> Path:
        return _get_shard_relative(file_name, suffix)

    def shard_local_path(self, file_name: str, suffix: str = "") -> Path:
        return self.local_cfg.base_path / self.shard_relative_path(file_name, suffix)

    def resolve_local(self, relative_path: Path) -> Path:
        return self.local_cfg.base_path / relative_path

    def mkdir(self, relative_path: Path) -> None:
        self.resolve_local(relative_path).parent.mkdir(parents=True, exist_ok=True)

    def collect_result(self, temp_path: Path, dest_path: Path, blocking: bool = True, callback: UploadCallBack = None):
        """Move a temp file into place locally and optionally upload to cloud."""
        relative_dest = dest_path
        if dest_path.is_absolute():
            try:
                relative_dest = dest_path.relative_to(self.local_cfg.base_path)
            except ValueError as exc:
                raise ValueError("Destination path must be inside local base path") from exc

        if self.cloud_cfg.enable and self.cloud is not None:
            self.cloud.upload(temp_path, self.cloud_cfg.base_path / relative_dest, blocking=blocking, callback=callback)

        if self.local_cfg.save_local:
            atomic_move(temp_path, self.local_cfg.base_path / relative_dest)
        else:
            unlink(temp_path) # TODO unlink only after successful upload, in case non blocking

    def exists_local(self, relative_path: Path) -> bool:
        return (self.local_cfg.base_path / relative_path).exists()

    def exists_cloud(self, relative_path: Path) -> bool:
        """Check if a relative path exists in cloud storage."""
        if self.cloud_cfg.enable and self.cloud is not None:
            return self.cloud.exists(self.cloud_cfg.base_path / relative_path)
        return False

    def cloud_upload(self, local_path: Path, relative_path: Path,
                     blocking: bool = True, callback: UploadCallBack = None):
        """Upload a local file to cloud at the given relative path."""
        if not self.cloud:
            raise ValueError("Cloud store is not enabled")
        self.cloud.upload(local_path, self.cloud_cfg.base_path / relative_path, blocking=blocking, callback=callback)

    def cloud_download(self, relative_path: Path, dest_local_path: Path):
        if not self.cloud:
            raise ValueError("Cloud store is not enabled")
        dest_local_path.parent.mkdir(parents=True, exist_ok=True)
        self.cloud.download(self.cloud_cfg.base_path / relative_path, dest_local_path)

    def load_data(self, path: Path, **kwargs):
        return self._LoadDataManager(self, path, **kwargs)

    def close(self):
        if self.cloud is not None:
            self.cloud.close()

    class _LoadDataManager:
        """Context manager to fetch a file from local or cloud, returning a file handle."""
        def __init__(self, file_manager_instance: FileManager, path: Path, **kwargs):
            self.fm_instance = file_manager_instance
            self.relative_path = path
            self.local_path: Path | None = None
            self.is_temporary: bool = False
            self.file_handle = None
            self.kwargs = kwargs

        def __enter__(self):
            if Path(self.relative_path).is_absolute():
                # Absolute paths are treated as local-only to avoid slow/broken S3 lookups.
                resolved_path = Path(self.relative_path)
                # Create parent dirs if the requested mode implies writing.
                mode = self.kwargs.get("mode", "")
                if any(flag in mode for flag in ("w", "a", "+")):
                    resolved_path.parent.mkdir(parents=True, exist_ok=True)
                self.is_temporary = False
            elif self.fm_instance.exists_local(self.relative_path):
                resolved_path = self.fm_instance.resolve_local(self.relative_path)
                self.is_temporary = False
            elif self.fm_instance.exists_cloud(self.relative_path):
                if self.fm_instance.local_cfg.cache_files:
                    resolved_path = self.fm_instance.resolve_local(self.relative_path)
                    resolved_path.parent.mkdir(parents=True, exist_ok=True)
                    self.fm_instance.cloud_download(self.relative_path, resolved_path)
                    self.is_temporary = False
                else:
                    resolved_path = self.fm_instance.create_temp_file(".s3_cached")
                    self.fm_instance.cloud_download(self.relative_path, resolved_path)
                    self.is_temporary = True
            else:
                raise FileNotFoundError(f"File not found locally or in cloud: {self.relative_path}")

            self.local_path = resolved_path
            self.file_handle = open(self.local_path, **self.kwargs)
            return self.file_handle

        def __exit__(self, exc_type, exc_value, traceback):
            if self.file_handle is not None:
                self.file_handle.close()
            if self.is_temporary and self.local_path:
                unlink(self.local_path)
