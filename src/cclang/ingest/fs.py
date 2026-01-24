"""Local + cloud file management utilities: sharded paths, temp files, and S3 uploads/downloads."""
from __future__ import annotations
import dataclasses
import hashlib
import os
import uuid
from os import unlink
from pathlib import Path
from typing import Optional

from cclang.config.s3 import S3Config
from cclang.io.cloud import S3Store, UploadCallBack
from cclang.ingest.exceptions import FileManagerError, LocalStorageError, LocalPathError
from cclang.ingest.exceptions import CloudStorageError, CloudNotConfiguredError, FileNotFoundInStorageError


def ensure_relative(path: Path, base: Path, label: str) -> Path:
    """Convert absolute path to relative to base; accept already-relative paths."""
    if path.is_absolute():
        try:
            return path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside data base path {base}") from exc
    if path.parts and path.parts[0] == base.name:
        return Path(*path.parts[1:])
    return path

def ensure_absolute(path: Path, base: Path) -> Path:
    """Return the absolute path rooted at base when input is relative."""
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == base.name:
        path = Path(*path.parts[1:])
    return base / path

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


def get_shard_relative(file_name: str, file_suffix: str) -> Path:
    shard_folder = file_name[:2].zfill(2)
    return Path(shard_folder) / (file_name + file_suffix)


def get_shard_path(base: Path | str, file_name: str, file_suffix: str) -> Path:
    """
    Public helper kept for backward compatibility.
    """
    return Path(base) / get_shard_relative(file_name, file_suffix)


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
        self.cloud: Optional[S3Store] = None
        if self.cloud_cfg.enable:
            self.cloud = S3Store(
                self.cloud_cfg.s3_config,
                self.cloud_cfg.max_upload_threads,
                max_pool_connections=self.cloud_cfg.max_pool_connections,
            )

    def _relative_to_local_base(self, path: Path) -> Path:
        """
        Make sure the path is relative to local_cfg.base_path.

        Absolute paths outside of base_path -> LocalPathError.
        Relative paths are returned as is.
        """
        path = Path(path)

        if path.is_absolute():
            try:
                return path.relative_to(self.local_cfg.base_path)
            except ValueError as exc:
                raise LocalPathError(
                    f"Destination path {path!r} must be inside local base path "
                    f"{self.local_cfg.base_path!r}"
                ) from exc

        return path

    def create_temp_file(self, suffix: str = "") -> Path:
        try:
            return _create_temp_file(self.local_cfg.temp_base, suffix)
        except OSError as exc:
            raise LocalStorageError(
                f"Failed to create temp file in {self.local_cfg.temp_base!s}"
            ) from exc

    def shard_local_path(self, file_name: str, suffix: str = "") -> Path:
        return self.local_cfg.base_path / get_shard_relative(file_name, suffix)

    def resolve_local(self, relative_path: Path) -> Path:
        return self.local_cfg.base_path / relative_path
    
    def resolve_cloud(self, relative_path: Path) -> Path:
        return self.cloud_cfg.base_path / relative_path

    def mkdir(self, path: str | Path, treat_as_file: Optional[bool] = None) -> Path:
        """
        Behavior:
        - If `treat_as_file` is None (default): the function tries to guess:
            * if the path has a suffix (e.g. ".txt", ".pdf") -> treated as file path
            * otherwise -> treated as directory path

        Returns:
            Path to the directory that was created (or already existed).
        """
        p = Path(path)

        if treat_as_file is True:
            # Force: always treat the path as a file path
            target_dir = p.parent
        elif treat_as_file is False:
            # Force: always treat the path as a directory path
            target_dir = p
        else:
            # Auto-detect: if there is a suffix, assume it's a file path
            # Otherwise assume it's a directory path
            target_dir = p if p.suffix == "" else p.parent

        # Create the directory and all missing parents; do nothing if it already exists
        target_dir.mkdir(parents=True, exist_ok=True)

        return target_dir

    def collect_result(
            self,
            temp_path: Path,
            dest_path: Path,
            blocking: bool = True,
            callback: UploadCallBack | None = None,
    ) -> Path:
        """
        Move a temp file into place locally and optionally upload it to the cloud.

        :return:
            - The final local path if save_local=True;
            - The path to the source file (temp_path or final_local_path),
              if save_local=False.

        :exception:
            - LocalStorageError / LocalPathError — problems with local filesystem;
            - CloudStorageError / CloudNotConfiguredError — problems with cloud storage.
        """
        temp_path = Path(temp_path)

        if not temp_path.exists():
            raise LocalStorageError(f"Temporary file does not exist: {temp_path!s}")

        if not self.local_cfg.save_local and not (self.cloud_cfg.enable and self.cloud):
            # Otherwise the result would just be lost.
            raise FileManagerError(
                "collect_result called with save_local=False while cloud is disabled: "
                "result would be lost."
            )

        relative_dest = self._relative_to_local_base(dest_path)
        final_local_path = self.local_cfg.base_path / relative_dest

        # 1. Local move (if enabled)
        if self.local_cfg.save_local:
            try:
                final_local_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise LocalStorageError(
                    f"Failed to create directory for {final_local_path!s}"
                ) from exc

            try:
                atomic_move(temp_path, final_local_path)
            except OSError as exc:
                raise LocalStorageError(
                    f"Failed to move temp file {temp_path!s} to {final_local_path!s}"
                ) from exc

            source_for_upload = final_local_path
        else:
            # We do not keep a local copy — the source for upload remains the temp file.
            source_for_upload = temp_path

        # 2. Upload to cloud (if configured)
        if self.cloud_cfg.enable:
            if not self.cloud:
                # This is possible if S3Store initialization failed while enable=True.
                raise CloudNotConfiguredError(
                    "Cloud config is enabled but S3Store is not initialized"
                )

            cloud_key = self.cloud_cfg.base_path / relative_dest
            try:
                self.cloud.upload(
                    source_for_upload,
                    cloud_key,
                    blocking=blocking,
                    callback=callback,
                )
            except Exception as exc:  # noqa: BLE001
                # The local file is already in place (if save_local=True),
                # so we do not touch it, just signal an S3 error.
                raise CloudStorageError(
                    f"Failed to upload {source_for_upload!s} to cloud key {cloud_key!s}"
                ) from exc

        # 3. Remove the temporary file if it is no longer needed.
        if not self.local_cfg.save_local:
            # For non-blocking uploads it is safer not to delete the file here,
            # but if blocking=True we can safely delete it.
            if blocking:
                try:
                    unlink(temp_path)
                except FileNotFoundError:
                    # The file was already removed by someone else — that's fine.
                    pass
                except OSError as exc:
                    raise LocalStorageError(
                        f"Failed to remove temporary file {temp_path!s}"
                    ) from exc

        return final_local_path if self.local_cfg.save_local else source_for_upload

    def exists_local(self, relative_path: Path) -> bool:
        return (self.local_cfg.base_path / relative_path).exists()

    def exists_cloud(self, relative_path: Path) -> bool:
        """Check if a relative path exists in cloud storage."""
        if self.cloud_cfg.enable and self.cloud is not None:
            try:
                return self.cloud.exists(self.resolve_cloud(relative_path))
            except Exception as exc:  # noqa: BLE001
                # exists() is usually non-critical; if needed, this can be logged/propagated.
                raise CloudStorageError(
                    f"Cloud exists() failed for {relative_path!s}"
                ) from exc
        return False

    def cloud_upload(
            self,
            local_path: Path,
            relative_path: Path,
            blocking: bool = True,
            callback: UploadCallBack | None = None,
    ) -> None:
        """Upload a local file to the cloud at the given relative path."""
        if not self.cloud_cfg.enable or not self.cloud:
            raise CloudNotConfiguredError("Cloud store is not enabled")

        try:
            self.cloud.upload(
                local_path,
                self.resolve_cloud(relative_path),
                blocking=blocking,
                callback=callback,
                )
        except Exception as exc:  # noqa: BLE001
            raise CloudStorageError(
                f"Failed to upload {local_path!s} to cloud path "
                f"{(self.resolve_cloud(relative_path))!s}"
            ) from exc

    def fetch_from_cloud(self, relative_path: Path, overwrite: bool = False) -> Path:
        """
        Download a file from cloud storage into the local data root.

        When overwrite=False (default), an existing local file is left untouched.
        """
        if not self.cloud_cfg.enable or not self.cloud:
            raise CloudNotConfiguredError("Cloud store is not enabled")

        relative_dest = ensure_relative(Path(relative_path), self.local_cfg.base_path, "relative_path")
        orig_cloud_path = self.resolve_cloud(relative_dest)
        dest_local_path = self.resolve_local(relative_dest)

        if dest_local_path.exists() and not overwrite:
            return dest_local_path

        try:
            dest_local_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LocalStorageError(
                f"Failed to create directory for {dest_local_path!s}"
            ) from exc

        try:
            self.cloud.download(
                orig_cloud_path,
                dest_local_path,
                )
        except Exception as exc:  # noqa: BLE001
            raise CloudStorageError(
                f"Failed to download cloud path "
                f"{(self.resolve_cloud(relative_dest))!s} "
                f"to {dest_local_path!s}"
            ) from exc
        return dest_local_path


    def cloud_download(self, relative_path: Path, dest_local_path: Path) -> None:
        """Download a cloud file at relative_path to dest_local_path locally."""
        if not self.cloud_cfg.enable or not self.cloud:
            raise CloudNotConfiguredError("Cloud store is not enabled")

        dest_local_path = Path(dest_local_path)
        try:
            dest_local_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LocalStorageError(
                f"Failed to create directory for {dest_local_path!s}"
            ) from exc

        try:
            self.cloud.download(
                self.resolve_cloud(relative_path),
                dest_local_path,
                )
        except Exception as exc:  # noqa: BLE001
            raise CloudStorageError(
                f"Failed to download cloud path "
                f"{(self.resolve_cloud(relative_path))!s} "
                f"to {dest_local_path!s}"
            ) from exc

    def load_data(self, path: Path, **kwargs):
        """
        Context manager to fetch a file from local or cloud, returning a file handle.

        Example:
            with fm.load_data(Path("corpora/xx/raw/foo.jsonl.zst"), mode="rb") as f:
                ...
        """
        return self._LoadDataManager(self, path, **kwargs)

    def close(self) -> None:
        if self.cloud is not None:
            self.cloud.close()

    class _LoadDataManager:
        """Context manager to fetch a file from local or cloud, returning a file handle."""

        def __init__(self, file_manager_instance: "FileManager", path: Path, **kwargs):
            self.fm_instance = file_manager_instance
            self.relative_path = path
            self.local_path: Optional[Path] = None
            self.is_temporary: bool = False
            self.file_handle = None
            self.kwargs = kwargs
            mode = self.kwargs.get("mode", "")
            self.is_write_mode = any(flag in mode for flag in ("w", "a", "+", "x"))

        def _resolve_path(self) -> Path:
            # Normalize Windows-style separators for relative paths before resolution.
            raw_path = str(self.relative_path)
            if not Path(raw_path).is_absolute():
                raw_path = raw_path.replace("\\", "/")
            path = Path(raw_path)
            if self.is_write_mode and not self.fm_instance.local_cfg.save_local and not self.fm_instance.cloud_cfg.enable:
                raise FileManagerError(
                    "Writing is not allowed when save_local=False and cloud is disabled; "
                    "result would be lost."
                )

            # 1. Absolute path — always local, no S3 involved.
            if path.is_absolute():
                if self.is_write_mode:
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        raise LocalStorageError(
                            f"Failed to create directory for {path!s}"
                        ) from exc
                self.is_temporary = False
                return path

            # 2. Relative path — first try to find it locally.
            if self.fm_instance.exists_local(path):
                self.is_temporary = False
                return self.fm_instance.resolve_local(path)

            # 3. If not present locally — try to fetch from cloud.
            if self.fm_instance.exists_cloud(path):
                if self.fm_instance.local_cfg.cache_files:
                    # Download into the local cache under base_path.
                    resolved_path = self.fm_instance.resolve_local(path)
                    try:
                        resolved_path.parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        raise LocalStorageError(
                            f"Failed to create directory for {resolved_path!s}"
                        ) from exc

                    self.fm_instance.cloud_download(path, resolved_path)
                    self.is_temporary = False
                    return resolved_path
                else:
                    # Download into a temporary file.
                    resolved_path = self.fm_instance.create_temp_file(".s3_cached")
                    self.fm_instance.cloud_download(path, resolved_path)
                    self.is_temporary = True
                    return resolved_path

            # 4. Not found locally or in the cloud.
            if self.is_write_mode:
                resolved_path = self.fm_instance.resolve_local(path)
                try:
                    resolved_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise LocalStorageError(
                        f"Failed to create directory for {resolved_path!s}"
                    ) from exc
                self.is_temporary = False
                return resolved_path
            raise FileNotFoundInStorageError(
                f"File not found locally or in cloud: {self.relative_path!s}"
            )

        def __enter__(self):
            resolved_path = self._resolve_path()
            self.local_path = resolved_path

            try:
                self.file_handle = open(self.local_path, **self.kwargs)
            except OSError as exc:
                raise LocalStorageError(
                    f"Failed to open file {self.local_path!s} with args {self.kwargs!r}"
                ) from exc

            return self.file_handle

        def __exit__(self, exc_type, exc_value, traceback):
            if self.file_handle is not None:
                try:
                    self.file_handle.close()
                except OSError:
                    # Closing usually should not fail; if it does, we do not
                    # override the original exception from the context.
                    pass

            if self.is_temporary and self.local_path:
                try:
                    unlink(self.local_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    # Deletion failure here is non-critical, so we ignore it.
                    pass

            if exc_type is None and self.is_write_mode:
                # Optional cloud upload for newly written relative files
                try:
                    if (
                        self.fm_instance.cloud_cfg.enable
                        and self.fm_instance.cloud
                        and not Path(self.relative_path).is_absolute()
                    ):
                        relative_dest = self.fm_instance._relative_to_local_base(Path(self.relative_path))
                        self.fm_instance.cloud_upload(
                            self.local_path,
                            relative_dest,
                            blocking=True,
                        )
                        if not self.fm_instance.local_cfg.save_local and self.local_path:
                            try:
                                unlink(self.local_path)
                            except FileNotFoundError:
                                pass
                            except OSError:
                                pass
                    elif not self.fm_instance.local_cfg.save_local:
                        # save_local=False but cloud disabled or absolute path — would lose data
                        raise FileManagerError(
                            "Writing with save_local=False requires cloud storage and relative path"
                        )
                except Exception:
                    # Propagate cloud/cleanup errors to the caller
                    raise
            # Do not suppress exceptions
            return False
