import hashlib
import os
import uuid
from pathlib import Path


def create_temp_file(base_path: Path | str, suff: str) -> Path:
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    name = str(uuid.uuid4()) + suff
    return base_path / name

def calculate_sha256(path: Path | str) -> str:
    path = Path(path)
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()

def get_shard_path(base: Path | str, file_name: str, file_suffix: str) -> Path:
    base = Path(base)
    shard_folder = file_name[:2].zfill(2)
    return base / shard_folder / (file_name + file_suffix)

def atomic_move(temp_file_path: Path | str, dest_path: Path | str) -> None:
    '''
    atomic move temp downloaded file to shard folder
    :param temp_file_path:
    :param dest_path:
    :return:
    '''
    dest_path = Path(dest_path)
    temp_file_path = Path(temp_file_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_file_path, dest_path)
