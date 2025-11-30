"""SQLite-backed store that tracks fetched items (url, sha256, local path, timestamp)."""
from datetime import datetime
import sqlite3
import threading
from pydantic import HttpUrl

from cclang.io.schemas import FetchedItem


def _fetched_item_from_db_resp(resp: sqlite3.Row) -> FetchedItem:
    keys = ['url', 'sha256', 'local_path', 'ts']
    resp_dict = dict(zip(keys, resp))
    return FetchedItem(**resp_dict)


class FetchedItemsStore:
    """Thread-safe registry of fetched items to avoid duplicate downloads."""
    def __init__(self, conn: sqlite3.Connection):
        self._db_conn = conn
        self._lock = threading.Lock()

    def get_url(self, url: str | HttpUrl) -> FetchedItem:
        """Return a fetched item by URL or None if absent."""
        url = str(url)
        with self._lock:
            if self._db_conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._db_conn.execute(
                "SELECT url, sha256, local_path, ts FROM fetch_items WHERE url = ?",
                (url,),
            )
            response_row = cur.fetchone()
            return _fetched_item_from_db_resp(response_row) if response_row else None

    def get_sha256(self, sha256: str) -> FetchedItem:
        """Return a fetched item by sha256 or None if absent."""
        with self._lock:
            if self._db_conn is None:
                raise RuntimeError('invalid acces to closed connection')

            cur = self._db_conn.execute(
                "SELECT url, sha256, local_path, ts FROM fetch_items WHERE sha256 = ?",
                (sha256,),
            )
            response_row = cur.fetchone()
            return _fetched_item_from_db_resp(response_row) if response_row else None

    def update_fetch_item(self, url: str | HttpUrl, sha256: str, local_path: str, ts: str | None = None):
        """Insert a fetched item record into SQLite."""
        url = str(url)
        ts = ts or datetime.now().isoformat() + 'Z'
        with self._lock:
            if self._db_conn is None:
                raise RuntimeError('invalid acces to closed connection')

            self._db_conn.execute(
                "INSERT INTO fetch_items (url, sha256, local_path, ts) VALUES (?, ?, ?, ?)",
                (url, sha256, local_path, ts),
            )
            self._db_conn.commit()

    def has_url(self, url: str) -> bool:
        return self.get_url(url) is not None

    def has_sha256(self, sha256: str) -> bool:
        return self.get_sha256(sha256) is not None

    def close(self):
        with self._lock:
            if self._db_conn is not None:
                try:
                    self._db_conn.close()
                    self._db_conn = None
                except Exception:
                    pass
