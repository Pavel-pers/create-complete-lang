import dataclasses
from os import unlink
from pathlib import Path
from typing import Optional, Type, TypeVar, IO

from pydantic import BaseModel

from cclang.config.s3 import S3Config
from cclang.core.manifest import ManifestStore
from cclang.io.exceptions import LocalStorageError
from cclang.io.fs import FileManager, LocalConfig, normalize_windows_path, ensure_relative
from cclang.io.cloud import S3Store, S3JobCallback
from cclang.io.schemas import UploadManifestRecord


class StorageManagerError(Exception):
    """Base exception for storage-related errors."""


ManifestRecordsType = TypeVar("T", bound=BaseModel)


@dataclasses.dataclass
class CloudConfig:
    enable: bool
    s3_config: S3Config
    max_upload_threads: int = 8
    max_download_threads: int = 0
    cloud_max_attempts: int = 4
    cloud_base_backoff: float = 0.5
    max_pool_connections: Optional[int] = None


class StorageManager:
    def __init__(self, local_cfg: LocalConfig, cloud_cfg: CloudConfig):
        # local setup
        self.local_cfg = local_cfg
        self._file_manager = FileManager(local_cfg)
        self._manifests: dict[str, ManifestStore[BaseModel]] = {}
        transfer_manifest = self._init_transfer_manifest()
        # cloud setup
        self._cloud = Optional[S3Store]
        self.cloud_cfg = cloud_cfg
        if self.cloud_cfg.enable:
            self._cloud = S3Store(
                cfg=self.cloud_cfg.s3_config,
                transfer_manifest=transfer_manifest,
                max_upload_threads=self.cloud_cfg.max_upload_threads,
                max_download_threads=self.cloud_cfg.max_download_threads,
                cloud_max_attempts=self.cloud_cfg.cloud_max_attempts,
                cloud_base_backoff=self.cloud_cfg.cloud_base_backoff,
                max_pool_connections=self.cloud_cfg.max_pool_connections
            )

    def _init_transfer_manifest(self) -> ManifestStore[UploadManifestRecord]:
        transfer_manifest = ManifestStore(
            Path('common/cloud_transfer_manifest.json'),
            UploadManifestRecord,
            self._file_manager,
            flush_every=1
        )
        self._manifests['_transfer_manifest'] = transfer_manifest
        return transfer_manifest

    def save_local_enabled(self) -> bool:
        return self.local_cfg.save_local

    def save_cloud_enabled(self) -> bool:
        return self.cloud_cfg.enable

    def register_manifest(self,
                          manifest_name: str | Path,
                          model_class: Type[ManifestRecordsType] = BaseModel,
                          flush_every: int = 1) -> ManifestStore[ManifestRecordsType]:
        manifest_path = self._file_manager.resolve_local(Path(manifest_name))
        manifest = ManifestStore(manifest_path, model_class, self._file_manager, flush_every=flush_every)
        self._manifests[str(manifest_name)] = manifest
        return manifest

    def push_data(self, data_path: Path, blocking=True) -> None:
        """
        Push local data to S3, override if exists.
        If call is non-blocking, not guaranteed that will succeed, cloud error will be reported in logs
        :param data_path: path to local data, realative or absolute in local base
        :param blocking: if True call is blocking until all data is pushed, otherwise it will return immediately
        :return:
        """
        relative_path = ensure_relative(data_path, self._file_manager.local_cfg.base_path, label='pushed data path')
        self._cloud.upload(self._file_manager.resolve_local(data_path), relative_key=relative_path, blocking=blocking)

    def push_manifest(self, manifest_name: Optional[str] = None):
        """
        Publish manifests to cloud, override if exists.
        :param manifest_name: manigest name if None push all manifests
        :return: None
        """
        if manifest_name is None:
            for manifest_name, manifest in self._manifests.items():
                manifest_relative_path = manifest.manifest_rel_path
                manifest_path = self._file_manager.resolve_local(manifest_relative_path)
                self.push_data(manifest_path, blocking=True)
        else:
            if manifest_name not in self._manifests:
                raise StorageManagerError(f"Manifest '{manifest_name}' does not exist")
            manifest = self._manifests.get(manifest_name)
            manifest_relative_path = manifest.manifest_rel_path
            manifest_path = self._file_manager.resolve_local(manifest_relative_path)
            self.push_data(manifest_path, blocking=True)

    class _LoadDataManager:
        def __init__(self, storage_manager: "StorageManager", path: Path, **kwargs):
            self._sm: StorageManager = storage_manager
            self._fm: FileManager = storage_manager._file_manager
            self._s3: S3Store = storage_manager._cloud
            self._relative_path = path
            self._local_path: Optional[Path] = None
            self._file_handle = None
            self._kwargs = kwargs
            mode = kwargs.get("mode", "")
            self._is_write_mode = any(flag in mode for flag in ("w", "a", "+", "x"))
            self._is_temporary = False

        def _ensure_local(self) -> Path:
            """
            Create local copy if not exists.
            :return: Local path of copy
            """
            path = normalize_windows_path(Path(self._relative_path))
            if self._is_write_mode and not (self._sm.save_cloud_enabled() or self._sm.save_local_enabled()):
                raise StorageManagerError('Writting is not save, because data will be lost;'
                                          'Because of save_local = False and s3 cloud is turned off')

            # absolute means that calling side is excepted that we take local file
            # * Search only in local in this case
            if path.is_absolute():
                if self._is_write_mode:
                    try:
                        self._fm.ensure_file(path)
                    except OSError as exc:
                        raise LocalStorageError(f"Failed to create file '{path}'") from exc

                self._is_temporary = False
                return path

            # * Try to find file localy
            if self._sm._exists_local(path):
                self._is_temporary = False
                return self._fm.resolve_local(path)

            # * Try to fetch from cloud
            if self._sm._exists_cloud(path):
                if self._fm.local_cfg.cache_files:
                    data_path = self._fm.resolve_local(path)
                    try:
                        data_path.parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        raise LocalStorageError(f"Failed to create dir '{path.parent}'") from exc
                    self._s3.download(path, data_path, blocking=True)
                    self._is_temporary = False
                    return data_path
                else:
                    data_path = self._fm.create_temp_file('.s3_cached')
                    self._s3.download(path, data_path, blocking=True)
                    self._is_temporary = True
                    return data_path

            # * File don't exist neither localy nor cloud
            if self._is_write_mode:
                data_path = self._fm.resolve_local(path)
                try:
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise LocalStorageError(f"Failed to create file '{path}'") from exc

                self._is_temporary = False
                return data_path

            raise FileNotFoundError(f"File not found neither localy nor in cloud: '{path}'")

        def __enter__(self) -> IO:
            data_path = self._ensure_local()
            self._local_path = data_path

            try:
                self._file_handle = open(data_path, **self._kwargs)
            except OSError as exc:
                raise LocalStorageError(f"Failed to open file '{data_path}'") from exc

            return self._file_handle

        def __exit__(self, exc_type, exc_val, exc_tb):
            # * Try to close file
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except OSError as exc:
                    # \-(:/)-/
                    pass
                self._file_handle = None

            # * Delete if temporary
            if self._is_temporary and self._local_path:
                try:
                    unlink(self._local_path)
                except FileNotFoundError:
                    # this is a good case, file not leaking
                    pass
                except OSError as exc:
                    # this is a bad case, memory leaking, maybe log it in future
                    pass

            # * Save changes to the cloud
            if exc_type is not None and self._is_write_mode:
                if self._sm.save_cloud_enabled():
                    self._sm.push_data(self._local_path, blocking=True)
                    if not self._sm.save_local_enabled():
                        try:
                            unlink(self._local_path)
                        except FileNotFoundError:
                            # this is a good case, file not leaking
                            pass
                        except OSError as exc:
                            # this is a bad case, memory is leaking, maybe log it in future
                            pass

            return False

    def open(self, path: Path, **kwargs) -> _LoadDataManager:
        return self._LoadDataManager(self, path, **kwargs)

    def finalize_artifact(
            self,
            temp_path: Path,
            dest_path: Path,
            blocking: bool = True,
            callback: S3JobCallback | None = None,
    ) -> None:
        """
        Finalize artifact by moving it from temp to final location, optionally upload it to the cloud.
        If call is non-blocking, not guaranteed that will succeed, cloud error will be reported in logs
        :return:
        """
        raise RuntimeError('Not implemented yet')
        if not temp_path.exists():
            raise FileNotFoundError(f"Temp file {temp_path} does not exist")
        if not self.save_local_enabled() and not self.save_cloud_enabled():
            raise StorageManagerError('Writting is not save, because data will be lost;'
                                      'Because of save_local = False and s3 cloud is turned off')

        relative_path = ensure_relative(dest_path, self._file_manager.local_cfg.base_path, label='collecting data path')

        # * local move
        pass


def _exists_local(self, path: Path):
    self._file_manager.exists(path)


def _exists_cloud(self, key: Path):
    if self._cloud is None:
        return False
    return self._cloud.exists(key)


def exists(self, path: Path) -> bool:
    return self._exists_local(path) or self._exists_cloud(path)
