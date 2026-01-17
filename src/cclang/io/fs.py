"""Local + cloud file management utilities: sharded paths, temp files, and S3 uploads/downloads."""
from __future__ import annotations
import dataclasses
import hashlib
import os
import typing
import uuid
from pathlib import Path
from typing import Optional, TextIO

from cclang.io.exceptions import LocalStorageError

def normalize_windows_path(relative_path) -> Path:
    raw_path = str(relative_path)
    if not Path(raw_path).is_absolute():
        raw_path = raw_path.replace("\\", "/")
    return Path(raw_path)

def ensure_relative(path: Path, base: Path, label: str) -> Path:
    """Convert absolute path to relative to base; accept already-relative paths."""
    if path.is_absolute():
        try:
            return path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside data base path {base}") from exc
    return path

def ensure_absolute(path: Path, base: Path) -> Path:
    """Return the absolute path rooted at base when input is relative."""
    if path.is_absolute():
        return path
    return base / path

def create_temp_file(base_path: Path, suff: str) -> Path:
    base_path.mkdir(parents=True, exist_ok=True)
    name = str(uuid.uuid4()) + suff
    (base_path / name).touch()
    return base_path / name


def get_shard_relative(file_name: str, file_suffix: str) -> Path:
    shard_folder = file_name[:2].zfill(2)
    return Path(shard_folder) / (file_name + file_suffix)


def get_shard_path(base: Path, file_name: str, file_suffix: str) -> Path:
    """
    Public helper kept for backward compatibility.
    """
    return base / get_shard_relative(file_name, file_suffix)


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


class FileManager:
    """Handles sharded local paths and optional S3 mirroring."""

    def __init__(self, local_cfg: LocalConfig):
        self.local_cfg = local_cfg

    def _relative_to_local_base(self, path: Path) -> Path:
        """
        Make sure the path is relative to local_cfg.base_path.

        Absolute paths outside of base_path -> LocalPathError.
        Relative paths are returned as is.
        """
        return ensure_relative(path, self.local_cfg.base_path, "data file")

    def create_temp_file(self, suffix: str = "") -> Path:
        try:
            return create_temp_file(self.local_cfg.temp_base, suffix)
        except OSError as exc:
            raise LocalStorageError(
                f"Failed to create temp file in {self.local_cfg.temp_base!s}"
            ) from exc

    def shard_local_path(self, file_name: str, suffix: str = "") -> Path:
        return get_shard_path(self.local_cfg.base_path, file_name, suffix)

    def resolve_local(self, relative_path: Path) -> Path:
        if relative_path.is_relative_to(self.local_cfg.base_path):
            return relative_path
        final_path = (self.local_cfg.base_path / relative_path).resolve()
        if final_path.is_relative_to(self.local_cfg.base_path):
            return final_path
        else:
            raise ValueError(f"Path {relative_path} is not inside data base path {self.local_cfg.base_path}")

    def mkdir(self, relative_path: str | Path) -> Path:
        path = self.resolve_local(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def exists(self, relative_path: Path) -> bool:
        return self.resolve_local(relative_path).exists()

    def ensure_file(self, relative_path: Path) -> Path:
        local_path = self.resolve_local(relative_path)
        if not self.exists(relative_path):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.touch()
            return local_path
        return local_path

    def open(self, relative_path: Path, mode: str = "r", **kwargs) -> typing.IO:
        local_path = self.resolve_local(relative_path)
        is_write_mode = any(flag in mode for flag in ("w", "a", "+", "x"))
        if not is_write_mode and not self.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")
        return self.ensure_file(relative_path).open(mode=mode, **kwargs)

    def finalize_artifact(self, temp_path: Path, dest_path: Path):
        """
        Finalizes the artifact by moving a temporary file to the destination path. It first ensures
        that the parent directory of the destination path exists by creating it if necessary.
        Then, it performs an atomic move operation from the temporary file to the resolved
        destination path.

        :param temp_path: The *absolute* path to the temporary file to be moved.
        :param dest_path: The destination path where the artifact will be stored, of type Path.
        :return: None
        """
        try:
            self.mkdir(dest_path.parent)
        except OSError as exc:
            raise LocalStorageError(f"Failed to create dir '{dest_path.parent}'") from exc

        try:
            atomic_move(temp_path, self.resolve_local(dest_path))
        except OSError as exc:
            raise LocalStorageError(f"Failed to move file '{temp_path}' to '{dest_path}'") from exc


    def close(self):
        pass
