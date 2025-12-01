from collections.abc import Iterable, Mapping
import threading
from datetime import datetime
from typing import Any, Optional
import psycopg

from cclang.io.schemas import PdfState, ProcessingStatus


def _pdf_state_from_db_resp(resp: Mapping[str, Any] | tuple[Any, ...] | None) -> Optional[PdfState]:
    if resp is None:
        return None

    keys = [
        "pdf_sha",
        "pdf_path",
        "text_sha",
        "text_path",
        "text_status",
        "text_updated_at",
        "tokenize_sha",
        "tokenize_path",
        "tokenize_status",
        "tokenize_updated_at",
        "lemma_sha",
        "lemma_path",
        "lemma_status",
        "lemma_updated_at",
    ]
    resp_dict = dict(resp) if isinstance(resp, Mapping) else dict(zip(keys, resp))

    for ts_key in ("text_updated_at", "tokenize_updated_at", "lemma_updated_at"):
        ts_val = resp_dict.get(ts_key)
        if isinstance(ts_val, datetime):
            resp_dict[ts_key] = ts_val.isoformat()
    return PdfState(**resp_dict)


class PdfStateStore:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn
        self._lock = threading.Lock()

    def get_pdf_sha(self, pdf_sha: str) -> Optional[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
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
                       WHERE pdf_sha = %s""",
                    (pdf_sha,),
                )
                response_row = cur.fetchone()
                return _pdf_state_from_db_resp(response_row)

    def get_text_sha(self, text_sha: str) -> Optional[PdfState]:
        if self._conn is None:
            raise RuntimeError("invalid access to closed connection")
        with self._conn.cursor() as cur:
            cur.execute(
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
                   WHERE text_sha = %s""",
                (text_sha,),
            )
            response_row = cur.fetchone()
            return _pdf_state_from_db_resp(response_row)

    def sync_with_fetched_items(self):
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pdf_state (pdf_sha, pdf_path)
                    SELECT fi.sha256     AS pdf_sha,
                           fi.local_path AS pdf_path
                    FROM fetch_items AS fi
                    ON CONFLICT (pdf_sha) DO NOTHING;
                    """
                )
            self._conn.commit()

    def update_text_status(self, pdf_sha: str,
                           text_status: ProcessingStatus | None,
                           text_sha: str | None,
                           text_path: str | None,
                           ts: str | None) -> None:
        text_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pdf_state
                    SET text_status = %s,
                        text_sha    = %s,
                        text_path   = %s,
                        text_updated_at = %s
                    WHERE pdf_sha = %s
                    """,
                    (text_status, text_sha, text_path, text_update_at, pdf_sha),
                )
            self._conn.commit()

    def update_lemma_status(self, pdf_sha: str,
                            lemma_status: ProcessingStatus | None,
                            lemma_sha: str | None,
                            ts: str | None) -> None:
        lemma_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pdf_state
                    SET lemma_status = %s,
                        lemma_sha    = %s,
                        lemma_updated_at = %s
                    WHERE pdf_sha = %s
                    """,
                    (lemma_status, lemma_sha, lemma_update_at, pdf_sha),
                )
            self._conn.commit()

    def has_text_sha(self, text_sha: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS(SELECT 1 FROM pdf_state WHERE text_sha = %s)
                    """,
                    (text_sha,),
                )
                row = cur.fetchone()
                return bool(row[0]) if row else False

    def has_lemma_sha(self, lemma_sha: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS(SELECT 1 FROM pdf_state WHERE lemma_sha = %s)
                    """,
                    (lemma_sha,),
                )
                row = cur.fetchone()
                return bool(row[0]) if row else False

    def filter_by_text_status(self, text_status: ProcessingStatus | None) -> Iterable[PdfState]:
        # !without streaming, because of lifetime of cursor
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                if text_status is None:
                    cur.execute(
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
                        """
                    )
                else:
                    cur.execute(
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
                           WHERE text_status = %s
                        """,
                        (text_status,),
                    )
                return map(_pdf_state_from_db_resp, cur.fetchall())

    def filter_by_lemma_status(self, lemma_status: ProcessingStatus | None) -> Iterable[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
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
                       WHERE lemma_status = %s
                    """,
                    (lemma_status,),
                )
                return map(_pdf_state_from_db_resp, cur.fetchall())

    def update_tokenize_status(self, pdf_sha: str,
                               tokenize_status: ProcessingStatus | None,
                               tokenize_sha: str | None,
                               ts: str | None) -> None:
        tokenize_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pdf_state
                    SET tokenize_status = %s,
                        tokenize_sha    = %s,
                        tokenize_updated_at = %s
                    WHERE pdf_sha = %s
                    """,
                    (tokenize_status, tokenize_sha, tokenize_update_at, pdf_sha),
                )
            self._conn.commit()

    def has_tokenize_sha(self, tokenize_sha: str) -> bool:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS(SELECT 1 FROM pdf_state WHERE tokenize_sha = %s)
                    """,
                    (tokenize_sha,),
                )
                row = cur.fetchone()
                return bool(row[0]) if row else False

    def filter_by_tokenize_status(self, tokenize_status: ProcessingStatus | None) -> Iterable[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
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
                       WHERE tokenize_status = %s
                    """,
                    (tokenize_status,),
                )
                return map(_pdf_state_from_db_resp, cur.fetchall())

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    self._conn = None
                except Exception: # noqa BLE:001
                    pass
