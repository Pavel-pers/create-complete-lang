import json
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from cclang.core.manifest import ManifestStore, FileManager


class TestRecord(BaseModel):
    id: str = Field(default="")
    name: str
    value: int


@pytest.fixture
def base_path(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def manifest_path(tmp_path):
    return Path("manifests/items.jsonl")


@pytest.fixture
def file_manager(base_path, tmp_path):
    cfg = MagicMock()
    cfg.base_path = base_path
    cfg.temp_base = tmp_path / "temp"
    fm = FileManager(cfg)
    return fm


@pytest.fixture
def manifest_store(file_manager, manifest_path):
    return ManifestStore(manifest_path, TestRecord, file_manager, flush_every=2)


def test_initialization_creates_parent_dirs(file_manager, manifest_path, tmp_path):
    manifest_dir = tmp_path / "manifests"
    if manifest_dir.is_dir():
        manifest_dir.rmdir()  # Убедимся, что директория не существует

    with patch.object(file_manager, 'mkdir') as mock_mkdir:
        ManifestStore(manifest_path, TestRecord, file_manager)
        mock_mkdir.assert_called_once_with(manifest_path.parent)


def test_load_data_empty_file(file_manager, manifest_path):
    store = ManifestStore(manifest_path, TestRecord, file_manager)
    assert len(store._items) == 0


def test_load_data_existing_file(file_manager, manifest_path, tmp_path):
    # Создаем тестовый manifest файл
    manifest_file = file_manager.resolve_local(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {"id": "1", "name": "item1", "value": 10},
        {"id": "2", "name": "item2", "value": 20}
    ]

    with open(manifest_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    store = ManifestStore(manifest_path, TestRecord, file_manager)
    assert len(store._items) == 2
    assert store.get("1").name == "item1"
    assert store.get("2").value == 20


def test_mark_flushes_buffer(manifest_store):
    record1 = TestRecord(id="1", name="first", value=100)
    record2 = TestRecord(id="2", name="second", value=200)

    manifest_store.mark(record1)
    assert len(manifest_store._buffer) == 1
    assert len(manifest_store._items) == 1

    # Вторая запись должна вызвать flush из-за flush_every=2
    manifest_store.mark(record2)
    assert len(manifest_store._buffer) == 0
    assert len(manifest_store._items) == 2

    # Проверяем, что данные записались на диск
    with manifest_store._fm.open(manifest_store.manifest_rel_path, mode="r", encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "1"
        assert json.loads(lines[1])["id"] == "2"


def test_items_returns_copy_not_reference(manifest_store):
    record = TestRecord(id="1", name="test", value=42)
    manifest_store.mark(record)

    items_list = list(manifest_store.items())
    assert len(items_list) == 1

    # Проверяем, что возвращается копия, а не ссылка
    with manifest_store._lock:
        manifest_store._items.clear()

    assert len(items_list) == 1  # Должно остаться неизменным


def test_thread_safety_with_concurrent_writes(file_manager, manifest_path):
    store = ManifestStore(manifest_path, TestRecord, file_manager, flush_every=100)

    def writer_thread(start_id):
        for i in range(10):
            record_id = f"{start_id}_{i}"
            store.mark(TestRecord(id=record_id, name=f"item_{record_id}", value=i))
            time.sleep(0.01)

    threads = [
        threading.Thread(target=writer_thread, args=(1,)),
        threading.Thread(target=writer_thread, args=(2,)),
        threading.Thread(target=writer_thread, args=(3,))
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Проверяем целостность данных
    assert len(store._items) == 30
    assert len(store._buffer) <= 30  # Некоторые записи могут быть в буфере

    # Принудительно сбрасываем буфер
    store.flush()

    # Проверяем, что все записи корректно записаны в файл
    with store._fm.open(store.manifest_rel_path, mode="r", encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 30


def test_get_returns_correct_record(manifest_store):
    record = TestRecord(id="test_id", name="test_name", value=42)
    manifest_store.mark(record)

    retrieved = manifest_store.get("test_id")
    assert retrieved is not None
    assert retrieved.id == "test_id"
    assert retrieved.name == "test_name"
    assert retrieved.value == 42

    assert manifest_store.get("non_existent") is None


def test_flush_handles_io_errors(manifest_store):
    with patch.object(manifest_store._fm, 'open', side_effect=OSError("Disk full")):
        record = TestRecord(id="1", name="error_test", value=100)
        manifest_store.mark(record)

        with pytest.raises(OSError):
            manifest_store.flush()


def test_overwriting_records(file_manager, manifest_path):
    store = ManifestStore(manifest_path, TestRecord, file_manager, flush_every=1)

    # Записываем первую версию
    store.mark(TestRecord(id="same_id", name="first", value=10))

    # Перезаписываем
    store.mark(TestRecord(id="same_id", name="second", value=20))

    # Проверяем, что последняя версия сохранена в памяти
    assert store.get("same_id").name == "second"
    assert store.get("same_id").value == 20

    # Проверяем, что обе версии записаны в файл (append-only)
    store.flush()
    with store._fm.open(store.manifest_rel_path, mode="r", encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 2
        first_record = json.loads(lines[0])
        second_record = json.loads(lines[1])
        assert first_record["name"] == "first"
        assert second_record["name"] == "second"