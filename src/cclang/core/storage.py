from pathlib import Path
from typing import Optional

from cclang.io.fs import FileManager, LocalConfig, normalize_windows_path
from cclang.io.cloud import S3Store, CloudConfig, UploadCallBack

class StorageManagerError(Exception):
    """Base exception for storage-related errors."""

class StorageManager:
    def __init__(self, local_cfg: LocalConfig, cloud_cfg: CloudConfig):
        self.local_cfg = local_cfg
        self.cloud_cfg = cloud_cfg
        self._file_manager = FileManager(local_cfg)
        self._cloud = Optional[S3Store]
        if self.cloud_cfg.enable:
            self._cloud = S3Store(
                cfg=self.cloud_cfg.s3_config,
                max_upload_threads=self.cloud_cfg.max_upload_threads,
                max_pool_connections=self.cloud_cfg.max_pool_connections,
            )

    def save_local_enabled(self) -> bool:
        return self.local_cfg.save_local

    def save_cloud_enabled(self) -> bool:
        return self.cloud_cfg.enable

    class _LoadDataManager:
        def __init__(self, storage_manager: "StorageManager", path: Path, **kwargs):
            self._sm: StorageManager = storage_manager
            self._relative_path = path
            self._local_path: Optional[Path] = None
            self._file_handle = None
            self._kwargs = kwargs
            mode = kwargs.get("mode", "")
            self._is_write_mode = any(flag in mode for flag in ("w", "a", "+", "x"))
            self._is_temporary = False

        def _prepare_data(self) -> Path:
            raise RuntimeError("not implemented")
            path = self._normalize_windows_path()

            if self._is_write_mode and not (self._sm.save_local_enabled() or self._sm.save_cloud_enabled()):
                raise StorageManagerError("Writing is not allowed, cause the data will be lost; "
                                          "In case cloud storage is disabled and save_local is False.")

            path = normalize_windows_path(path)
            if path.is_absolute():
                if self._is_write_mode:
                    self._sm._file_manager.ensure_file(path)
                else:

                self._is_temporary = False
                return path

            if self._sm._exists_local(path):
                self._is_temporary = False
                return self._sm._file_manager.resolve_local(path)

            if self._exists_cloud(path):
                # TODO


    def collect_result(
            self,
            temp_path: Path,
            dest_path: Path,
            blocking: bool = True,
            callback: UploadCallBack | None = None,
     ) -> Path:
        raise RuntimeError("not implemented")



    def _exists_local(self, path: Path):
        self._file_manager.exists(path)

    def _exists_cloud(self, key: Path):
        if self._cloud is None:
            return False
        return self._cloud.exists(key)

    def exists(self, path: Path) -> bool:
        return self._exists_local(path) or self._exists_cloud(path)


