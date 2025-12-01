from collections.abc import Mapping
from datetime import datetime
import threading
from typing import Any
from pydantic import HttpUrl
import psycopg

from cclang.io.schemas import FetchedItem


def _fetched_item_from_db_resp(resp: Mapping[str, Any] | tuple[Any, ...] | None) -> FetchedItem | None:
    """Return a `FetchedItem` from a DB response row."""
    if resp is None:
        return None

    if isinstance(resp, Mapping):
        resp_dict = dict(resp)
    else:
        keys = ["url", "sha256", "local_path", "ts"]
        resp_dict = dict(zip(keys, resp))

    ts_val = resp_dict.get("ts")
    if isinstance(ts_val, datetime):
        resp_dict["ts"] = ts_val.isoformat()
    return FetchedItem(**resp_dict)


class FetchedItemsStore:
    def __init__(self, conn: psycopg.Connection):
        self._db_conn = conn
        self._lock = threading.Lock()

    def get_url(self, url: str | HttpUrl) -> FetchedItem | None:
        url = str(url)
        with self._lock:
            if self._db_conn is None:
                raise RuntimeError('invalid access to closed connection')
            with self._db_conn.cursor() as cur:
                cur.execute(
                    "SELECT url, sha256, local_path, ts FROM fetch_items WHERE url = %s",
                    (url,),
                )
                response_row = cur.fetchone()
                return _fetched_item_from_db_resp(response_row)

    def get_sha256(self, sha256: str) -> FetchedItem | None:
        with self._lock:
            if self._db_conn is None:
                raise RuntimeError('invalid access to closed connection')

            with self._db_conn.cursor() as cur:
                cur.execute(
                    "SELECT url, sha256, local_path, ts FROM fetch_items WHERE sha256 = %s",
                    (sha256,),
                )
                response_row = cur.fetchone()
                return _fetched_item_from_db_resp(response_row)

    def update_fetch_item(self, url: str | HttpUrl, sha256: str, local_path: str, ts: str | None = None):
        url = str(url)
        ts = ts or datetime.now().isoformat() + 'Z'
        with self._lock:
            if self._db_conn is None:
                raise RuntimeError('invalid access to closed connection')

            with self._db_conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO fetch_items (url, sha256, local_path, ts) VALUES (%s, %s, %s, %s)",
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
