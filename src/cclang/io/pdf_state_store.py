import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterable
import sqlite3

from cclang.io.schemas import PdfState, ProcessingStatus


def _pdf_state_from_db_resp(resp: sqlite3.Row) -> Optional[PdfState]:
    keys = ['pdf_sha',
            'pdf_path',
            'text_sha',
            'text_path',
            'text_status',
            'text_updated_at',
            'tokenize_sha',
            'tokenize_path',
            'tokenize_status',
            'tokenize_updated_at',
            'lemma_sha',
            'lemma_path',
            'lemma_status',
            'lemma_updated_at']
    resp_dict = dict(zip(keys, resp))
    return PdfState(**resp_dict)


class PdfStateStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.Lock()

    def get_pdf_sha(self, pdf_sha: str) -> Optional[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """SELECT pdf_sha,
                          pdf_path,
                          text_sha,
                          text_path,
                          text_status,
                          text_updated_at,
                          tokenize_sha,
                          tokenize_path,
                          tokenize_status,
                          tokenize_updated_at,
                          lemma_sha,
                          lemma_path,
                          lemma_status,
                          lemma_updated_at
                   FROM pdf_state
                   WHERE pdf_sha = ?""",
                (pdf_sha,)
            )
            response_row = cur.fetchone()
            return _pdf_state_from_db_resp(response_row) if response_row else None

    def get_text_sha(self, text_sha: str) -> Optional[PdfState]:
        if self._conn is None:
            raise RuntimeError('invalid acces to closed connection')
        cur = self._conn.execute(
            """SELECT pdf_sha,
                      pdf_path,
                      text_sha,
                      text_path,
                      text_status,
                      text_updated_at,
                      tokenize_sha,
                      tokenize_path,
                      tokenize_status,
                      tokenize_updated_at,
                      lemma_sha,
                      lemma_path,
                      lemma_status,
                      lemma_updated_at
               FROM pdf_state
               WHERE text_sha = ?""",
            (text_sha,)
        )
        response_row = cur.fetchone()
        return _pdf_state_from_db_resp(response_row) if response_row else None


    def sync_with_fetched_items(self):
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO pdf_state (pdf_sha, pdf_path)
                SELECT fi.sha256     AS pdf_sha,
                       fi.local_path AS pdf_path
                FROM fetch_items AS fi
                """)
            self._conn.commit()

    def update_text_status(self, pdf_sha: str,
                           text_status: ProcessingStatus | None,
                           text_sha: str | None,
                           text_path: str | None,
                           ts: str | None) -> None:
        text_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            self._conn.execute(
                """
                UPDATE pdf_state
                SET text_status = ?,
                    text_sha    = ?,
                    text_path   = ?,
                    text_updated_at = ?
                WHERE pdf_sha = ?
                """, (text_status, text_sha, text_path, text_update_at, pdf_sha)
            )
            self._conn.commit()

    def update_lemma_status(self, pdf_sha: str,
                            lemma_status: ProcessingStatus | None,
                            lemma_sha: str | None,
                            ts: str | None) -> None:
        lemma_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            self._conn.execute(
                """
                UPDATE pdf_state
                SET lemma_status = ?,
                    lemma_sha    = ?
                WHERE pdf_sha = ?
                """, (lemma_status, lemma_sha, pdf_sha)
            )
            self._conn.commit()

    def has_text_sha(self, text_sha: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """
                SELECT EXISTS(SELECT 1 FROM pdf_state WHERE text_sha = ?)
                LIMIT 1
                """, (text_sha,)
            )
            return cur.fetchone()[0]

    def has_lemma_sha(self, lemma_sha: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """
                SELECT EXISTS(SELECT 1 FROM pdf_state where lemma_sha = ?)
                LIMIT 1
                """, (lemma_sha,)
            )
            return cur.fetchone()[0]

    def filter_by_text_status(self, text_status: ProcessingStatus | None) -> Iterable[PdfState]:
        # !without streaming, because of lifetime of cursor
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            if text_status is None:
                cur = self._conn.execute(
                    """SELECT pdf_sha,
                              pdf_path,
                              text_sha,
                              text_path,
                              text_status,
                              text_updated_at,
                              tokenize_sha,
                              tokenize_path,
                              tokenize_status,
                              tokenize_updated_at,
                              lemma_sha,
                              lemma_path,
                              lemma_status,
                              lemma_updated_at
                       FROM pdf_state
                       WHERE text_status IS NULL
                    """)
            else:
                cur = self._conn.execute(
                    """SELECT pdf_sha,
                              pdf_path,
                              text_sha,
                              text_path,
                              text_status,
                              text_updated_at,
                              tokenize_sha,
                              tokenize_path,
                              tokenize_status,
                              tokenize_updated_at,
                              lemma_sha,
                              lemma_path,
                              lemma_status,
                              lemma_updated_at
                       FROM pdf_state
                       WHERE text_status = ?
                    """, (text_status,))
            return map(_pdf_state_from_db_resp, cur.fetchall())

    def filter_by_lemma_status(self, lemma_status: ProcessingStatus | None) -> Iterable[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """SELECT pdf_sha,
                          pdf_path,
                          text_sha,
                          text_path,
                          text_status,
                          text_updated_at,
                          tokenize_sha,
                          tokenize_path,
                          tokenize_status,
                          tokenize_updated_at,
                          lemma_sha,
                          lemma_path,
                          lemma_status,
                          lemma_updated_at
                   FROM pdf_state
                   WHERE lemma_status = ?
                """, (lemma_status,))
            return map(_pdf_state_from_db_resp, cur.fetchall())

    def update_tokenize_status(self, pdf_sha: str,
                               tokenize_status: ProcessingStatus | None,
                               tokenize_sha: str | None,
                               ts: str | None) -> None:
        tokenize_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            self._conn.execute(
                """
                UPDATE pdf_state
                SET tokenize_status = ?,
                    tokenize_sha    = ?
                WHERE pdf_sha = ?
                """, (tokenize_status, tokenize_sha, pdf_sha)
            )
            self._conn.commit()

    def has_tokenize_sha(self, tokenize_sha: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """
                SELECT EXISTS(SELECT 1 FROM pdf_state WHERE tokenize_sha = ?)
                LIMIT 1
                """, (tokenize_sha,)
            )
            return cur.fetchone()[0]

    def filter_by_tokenize_status(self, tokenize_status: ProcessingStatus | None) -> Iterable[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError('invalid acces to closed connection')
            cur = self._conn.execute(
                """SELECT pdf_sha,
                          pdf_path,
                          text_sha,
                          text_path,
                          text_status,
                          text_updated_at,
                          tokenize_sha,
                          tokenize_path,
                          tokenize_status,
                          tokenize_updated_at,
                          lemma_sha,
                          lemma_path,
                          lemma_status,
                          lemma_updated_at
                   FROM pdf_state
                   WHERE tokenize_status = ?
                """, (tokenize_status,))
            return map(_pdf_state_from_db_resp, cur.fetchall())

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    self._conn = None
                except Exception: # noqa BLE:001
                    pass
