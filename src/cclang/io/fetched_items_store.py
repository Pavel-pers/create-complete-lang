from datetime import datetime
from pathlib import Path
import sqlite3

from pydantic import HttpUrl

from cclang.io.schemas import FetchedItem
from cclang.io.db import get_conn


def _fetched_item_from_db_resp(resp: sqlite3.Row) -> FetchedItem:
    keys = ['url', 'sha256', 'local_path', 'ts']
    resp_dict = dict(zip(keys, resp))
    return FetchedItem(**resp_dict)


class FetchedItemsStore:
    def __init__(self, conn: sqlite3.Connection):
        self._db_conn = conn
        self._db_cur = conn.cursor()

    def get_url(self, url: str) -> FetchedItem:
        self._db_cur.execute("""
                    SELECT url, sha256, local_path, ts FROM fetch_items
                             WHERE url = ?""", (url, ))
        response_row = self._db_cur.fetchone()
        return _fetched_item_from_db_resp(response_row) if response_row else None

    def get_sha256(self, sha256: str) -> FetchedItem:
        self._db_cur.execute("""
                             SELECT url, sha256, local_path, ts FROM fetch_items
                             WHERE sha256 = ?""", (sha256, ))
        response_row = self._db_cur.fetchone()
        return _fetched_item_from_db_resp(response_row) if response_row else None

    def update_fetch_item(self, url: str, sha256: str, local_path: str, ts: str | None = None):
        ts = ts or datetime.now().isoformat() + 'Z'
        self._db_cur.execute("""
                    INSERT INTO fetch_items (url, sha256, local_path, ts) VALUES (?, ?, ?, ?)
        """, (url, sha256, local_path, ts))
        self._db_conn.commit()

    def has_url(self, url: str) -> bool:
        return self.get_url(url) is not None

    def has_sha256(self, sha256: str) -> bool:
        return self.get_sha256(sha256) is not None

    def close(self):
        self._db_cur.close()