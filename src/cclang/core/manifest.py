"""Thread-safe append-only manifest writer/reader for pipeline outputs."""
import threading
from pathlib import Path
from typing import TypeVar, Generic, Type, Optional
import json
from pydantic import BaseModel

from cclang.io.fs import FileManager

ManifestRecord = TypeVar("ManifestRecord")
class ManifestStore(Generic[ManifestRecord]):
    """Append-only JSONL manifest storage with in-memory index."""
    def __init__(self, manifest_rel_path: Path, model_class: Type[BaseModel], fm: FileManager, flush_every: int = 1):
        self._fm = fm
        self._manifest_rel_path: Path = manifest_rel_path
        self._fm.mkdir(self._manifest_rel_path.parent)
        self._items: dict = {}
        # RLock is required because mark() may call flush() under the same lock
        self._lock = threading.RLock()
        self._buffer: list[ManifestRecord] = []
        self._flush_every = max(1, flush_every)
        self._load_data(model_class)

    def _load_data(self, cls: Type[BaseModel]):
        if not self._fm.exists(self._manifest_rel_path):
            return

        with self._fm.open(self._manifest_rel_path, mode="r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                payload = json.loads(line)
                record = cls(**payload)
                self._items[record.id] = record

    @property
    def manifest_rel_path(self) -> Path:
        return self._manifest_rel_path

    def flush(self):
        """Append buffered records to disk."""
        if not self._buffer:
            return

        with self._lock:
            # Ensure manifest file exists locally so FileManager can open it for append.
            resolved_manifest_path = self._fm.resolve_local(self._manifest_rel_path)
            self._fm.mkdir(self._manifest_rel_path.parent)
            resolved_manifest_path.touch(exist_ok=True)
            with self._fm.open(self._manifest_rel_path, mode="a", encoding='utf-8') as f:
                for record in self._buffer:
                    payload = record.model_dump(mode="json")
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._buffer.clear()

    def mark(self, record: ManifestRecord):
        """Add record to buffer and flush based on flush_every."""
        with self._lock:
            self._items[record.id] = record
            self._buffer.append(record)
            if len(self._buffer) >= self._flush_every:
                self.flush()

    def items(self):
        return self._items.values()

    def get(self, record_id)->ManifestRecord:
        return self._items.get(record_id)

    def __del__(self):
        try:
            self.flush()
        except Exception: # noqa BLE:001
            # \--(:/)--/
            pass
