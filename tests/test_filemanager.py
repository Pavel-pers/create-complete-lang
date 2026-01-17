import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from cclang.io.fs import (
    ensure_relative,
    ensure_absolute,
    create_temp_file,
    get_shard_relative,
    get_shard_path,
    calculate_sha256,
    atomic_move,
    LocalConfig,
    FileManager,
    LocalStorageError,
)
from cclang.io.exceptions import LocalStorageError as OriginalLocalStorageError

@pytest.fixture
def base_path(tmp_path):
    return tmp_path / "data_base"

@pytest.fixture
def temp_base(tmp_path):
    return tmp_path / "temp"

@pytest.fixture
def local_cfg(base_path, temp_base):
    return LocalConfig(
        base_path=base_path,
        save_local=True,
        cache_files=True,
        temp_base=temp_base
    )

@pytest.fixture
def file_manager(local_cfg):
    return FileManager(local_cfg)

# --- Тесты для вспомогательных функций ---
def test_ensure_relative_inside(base_path):
    target = base_path / "subdir/file.txt"
    assert ensure_relative(target, base_path, "test") == Path("subdir/file.txt")

def test_ensure_relative_outside(base_path):
    target = Path("/outside/file.txt")
    with pytest.raises(ValueError, match="must be inside data base path"):
        ensure_relative(target, base_path, "test")

def test_ensure_absolute_simple(base_path):
    rel_path = Path("subdir/file.txt")
    assert ensure_absolute(rel_path, base_path) == base_path / "subdir/file.txt"

def test_ensure_absolute_already_absolute(base_path):
    abs_path = base_path / "file.txt"
    assert ensure_absolute(abs_path, base_path) == abs_path

def test_create_temp_file(temp_base):
    temp_base.mkdir()
    temp_path = create_temp_file(temp_base, ".tmp")
    assert temp_path.parent == temp_base
    assert temp_path.suffix == ".tmp"
    assert temp_path.exists()  # Файл не создается!

def test_get_shard_relative():
    assert get_shard_relative("abc123", ".bin") == Path("ab/abc123.bin")

def test_get_shard_path(base_path):
    assert get_shard_path(base_path, "xyz789", ".dat") == base_path / "xy/xyz789.dat"

def test_calculate_sha256(tmp_path):
    file_path = tmp_path / "test.bin"
    file_path.write_bytes(b"hello")
    assert calculate_sha256(file_path) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

def test_atomic_move(tmp_path):
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst" / "file.txt"
    src.write_text("content")
    atomic_move(src, dst)
    assert dst.read_text() == "content"
    assert not src.exists()

# --- Тесты для FileManager ---
def test_shard_local_path(file_manager, base_path):
    path = file_manager.shard_local_path("item_42", ".json")
    assert path == base_path / "it/item_42.json"

def test_mkdir_creates_directory(file_manager, base_path):
    dir_path = Path("new/dir")
    created_path = file_manager.mkdir(dir_path)
    assert created_path == base_path / "new/dir"
    assert created_path.is_dir()

def test_exists(file_manager, base_path):
    file_path = Path("existing.txt")
    file_manager.ensure_file(file_path)
    assert file_manager.exists(file_path) is True
    assert file_manager.exists(Path("nonexistent.txt")) is False

def test_ensure_file_creates_missing(file_manager, base_path):
    file_path = Path("new_file.txt")
    result = file_manager.ensure_file(file_path)
    assert result == base_path / "new_file.txt"
    assert result.exists()

def test_ensure_file_returns_existing(file_manager, base_path):
    existing = base_path / "old.txt"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("data")
    result = file_manager.ensure_file(Path("old.txt"))
    assert result == existing

def test_open_read_mode(file_manager, base_path):
    file_path = Path("readme.txt")
    (base_path / file_path).parent.mkdir(parents=True, exist_ok=True)
    (base_path / file_path).write_text("content")
    with file_manager.open(file_path, "r") as f:
        assert f.read() == "content"

def test_open_write_mode_creates_parents(file_manager, base_path):
    file_path = Path("subdir/new.txt")
    with file_manager.open(file_path, "w") as f:
        f.write("data")
    assert (base_path / file_path).read_text() == "data"
    assert (base_path / "subdir").is_dir()

def test_open_fails_for_missing_file_read_mode(file_manager):
    with pytest.raises(FileNotFoundError, match="File not found"):
        file_manager.open(Path("missing.txt"), "r")

def test_finalize_artifact(file_manager, base_path, temp_base):
    temp_base.mkdir()
    temp_file = temp_base / "temp.tmp"
    temp_file.write_text("artifact data")
    dest_path = Path("artifacts/model.bin")

    file_manager.finalize_artifact(temp_file, dest_path)

    final_path = base_path / dest_path
    assert final_path.read_text() == "artifact data"
    assert not temp_file.exists()
    assert (base_path / "artifacts").is_dir()

# --- Тесты безопасности ---
def test_resolve_local_prevents_traversal(file_manager, base_path):
    malicious_path = Path("../../etc/passwd")
    # Проверка через публичный метод
    with pytest.raises(ValueError):
        file_manager.resolve_local(malicious_path)

# --- Тесты ошибок ---
def test_create_temp_file_fails_if_no_permission(file_manager, tmp_path):
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir(mode=0o444)  # Только чтение
    file_manager.local_cfg.temp_base = protected_dir

    with pytest.raises(LocalStorageError):
        file_manager.create_temp_file(".tmp")


# --- Edge cases ---
def test_ensure_relative_accepts_relative_path(base_path):
    rel_path = Path("dir/file.txt")
    assert ensure_relative(rel_path, base_path, "test") == rel_path

def test_ensure_absolute_handles_dot(base_path):
    assert ensure_absolute(Path("."), base_path) == base_path