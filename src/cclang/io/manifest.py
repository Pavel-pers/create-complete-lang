import threading
from pathlib import Path
from typing import TypeVar, Generic, Type
import json
from pydantic import BaseModel

ManifestRecord = TypeVar("ManifestRecord")
class ManifestStore(Generic[ManifestRecord]):
    def __init__(self, manifest_path: Path, model_class: Type[BaseModel], flush_every: int = 1):
        self._manifest_path: Path = manifest_path
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self._items: dict = {}
        # RLock нужен, потому что mark() может вызвать flush() под тем же локом
        self._lock = threading.RLock()
        self._buffer: list[ManifestRecord] = []
        self._flush_every = max(1, flush_every)
        self._load_data(model_class)

    def _load_data(self, cls: Type[BaseModel]):
        if not self._manifest_path.exists():
            return

        with open(self._manifest_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                payload = json.loads(line)
                record = cls(**payload)
                self._items[record.id] = record

    def flush(self):
        if not self._buffer:
            return

        with self._lock:
            with open(self._manifest_path, "a", encoding='utf-8') as f:
                for record in self._buffer:
                    payload = record.model_dump(mode="json")
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._buffer.clear()

    def mark(self, record: ManifestRecord):
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
