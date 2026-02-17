from collections.abc import Iterable, Mapping
import threading
from datetime import datetime
from typing import Any, Optional, List
import psycopg
from psycopg import sql

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
    for status_key in ("text_status", "tokenize_status", "lemma_status"):
        if resp_dict.get(status_key) is None:
            resp_dict[status_key] = ProcessingStatus.NOT_PROCESSED
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
                           text_status: ProcessingStatus,
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
                    SET text_status     = %s,
                        text_sha        = %s,
                        text_path       = %s,
                        text_updated_at = %s
                    WHERE pdf_sha = %s
                    """,
                    (text_status.value, text_sha, text_path, text_update_at, pdf_sha),
                )
            self._conn.commit()

    def update_lemma_status(self, pdf_sha: str,
                            lemma_status: ProcessingStatus,
                            lemma_sha: str | None,
                            lemma_path: str | None,
                            ts: str | None) -> None:
        lemma_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pdf_state
                    SET lemma_status     = %s,
                        lemma_sha        = %s,
                        lemma_path       = %s,
                        lemma_updated_at = %s
                    WHERE pdf_sha = %s
                    """,
                    (lemma_status.value, lemma_sha, lemma_path, lemma_update_at, pdf_sha),
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

    def filter_by_text_status(self, text_status: ProcessingStatus) -> Iterable[PdfState]:
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
                       WHERE text_status = %s
                    """,
                    (text_status.value,),
                )
                return map(_pdf_state_from_db_resp, cur.fetchall())

    def filter_by_lemma_status(self, lemma_status: ProcessingStatus) -> Iterable[PdfState]:
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
                    (lemma_status.value,),
                )
                return map(_pdf_state_from_db_resp, cur.fetchall())

    def update_tokenize_status(self, pdf_sha: str,
                               tokenize_status: ProcessingStatus,
                               tokenize_sha: str | None,
                               tokenize_path: str | None,
                               ts: str | None) -> None:
        tokenize_update_at: Optional[str] = ts
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pdf_state
                    SET tokenize_status     = %s,
                        tokenize_sha        = %s,
                        tokenize_path       = %s,
                        tokenize_updated_at = %s
                    WHERE pdf_sha = %s
                    """,
                    (tokenize_status.value, tokenize_sha, tokenize_path, tokenize_update_at, pdf_sha),
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

    def filter_by_tokenize_status(self, tokenize_status: ProcessingStatus) -> Iterable[PdfState]:
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
                    (tokenize_status.value, ),
                )
                return map(_pdf_state_from_db_resp, cur.fetchall())

    def filter_by_status(self,
                         text_status: ProcessingStatus | None = None,
                         tokenize_status: ProcessingStatus | None = None,
                         lemma_status: ProcessingStatus | None = None) -> Iterable[PdfState]:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            cond_query: List[str]= []
            cond_values: List[str]= []

            if text_status is not None:
                cond_query.append("text_status = %s")
                cond_values.append(text_status.value)
            if tokenize_status is not None:
                cond_query.append("tokenize_status = %s")
                cond_values.append(tokenize_status.value)
            if lemma_status is not None:
                cond_query.append("lemma_status = %s")
                cond_values.append(lemma_status.value)

            select_clause = sql.SQL("""
                    SELECT pdf_sha,
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
                    FROM pdf_state""")

            if cond_query:
                cond_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(map(sql.SQL, cond_query))
            else:
                cond_clause = sql.SQL("")

            query = select_clause + cond_clause

            with self._conn.cursor() as cur:
                cur.execute(query, cond_values)
                return map(_pdf_state_from_db_resp, cur.fetchall())


    # ── lemma_results table (multi-lemmatizer support) ──────────────

    def ensure_lemma_results_table(self):
        """Ensure lemma_results exists with composite PK (doc_id, method)."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'lemma_results'
                    )
                """)
                table_exists = cur.fetchone()[0]

                if not table_exists:
                    cur.execute("""
                        CREATE TABLE lemma_results (
                            doc_id TEXT NOT NULL,
                            artefact_id TEXT,
                            status TEXT NOT NULL,
                            path TEXT,
                            method TEXT NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE,
                            PRIMARY KEY (doc_id, method)
                        )
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_lemma_results_status
                        ON lemma_results(method, status)
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_lemma_results_artefact_id
                        ON lemma_results(artefact_id)
                    """)
                else:
                    # Check if PK is already composite
                    cur.execute("""
                        SELECT count(*) FROM information_schema.key_column_usage
                        WHERE table_name = 'lemma_results'
                          AND constraint_name = 'lemma_results_pkey'
                    """)
                    pk_col_count = cur.fetchone()[0]
                    if pk_col_count == 1:
                        cur.execute(
                            "ALTER TABLE lemma_results DROP CONSTRAINT lemma_results_pkey"
                        )
                        cur.execute(
                            "ALTER TABLE lemma_results ADD PRIMARY KEY (doc_id, method)"
                        )
                        # Recreate index for (method, status) queries
                        cur.execute("DROP INDEX IF EXISTS idx_lemma_results_status")
                        cur.execute("""
                            CREATE INDEX idx_lemma_results_status
                            ON lemma_results(method, status)
                        """)
            self._conn.commit()

    def migrate_lemma_data_from_pdf_state(self):
        """Migrate existing apertium lemma data from pdf_state_old into lemma_results."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lemma_results
                        (doc_id, method, status, artefact_id, path, updated_at)
                    SELECT pdf_sha, 'apertium-mar-morph', lemma_status, lemma_sha,
                           lemma_path, lemma_updated_at::timestamptz
                    FROM pdf_state_old
                    WHERE lemma_status IS NOT NULL AND lemma_status != 'pending'
                    ON CONFLICT (doc_id, method) DO NOTHING
                """)
                migrated = cur.rowcount
            self._conn.commit()
            return migrated

    def upsert_lemma_result(self, doc_id: str, method: str,
                            status: str, artefact_id: str | None = None,
                            path: str | None = None,
                            ts: str | None = None) -> None:
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lemma_results
                        (doc_id, method, status, artefact_id, path, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, method) DO UPDATE SET
                        status      = EXCLUDED.status,
                        artefact_id = EXCLUDED.artefact_id,
                        path        = EXCLUDED.path,
                        updated_at  = EXCLUDED.updated_at
                """, (doc_id, method, status, artefact_id, path, ts))
            self._conn.commit()

    def filter_unprocessed_for_lemmatizer(
        self, lemmatizer: str, limit: int | None = None,
        include_processed: bool = False,
    ) -> List[PdfState]:
        """Return docs with tokenize=ok not yet processed by *lemmatizer*.

        When *include_processed* is True, return ALL tokenized docs regardless
        of existing lemma results (used for --target-status all).
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("invalid access to closed connection")

            base = """
                SELECT d.doc_id,
                       d.path,
                       o.artefact_id,
                       o.path,
                       o.status,
                       o.updated_at,
                       t.artefact_id,
                       t.path,
                       t.status,
                       t.updated_at,
                       NULL, NULL, 'pending', NULL
                FROM documents d
                JOIN ocr_results o      ON d.doc_id = o.doc_id AND o.status = 'ok'
                JOIN tokenize_results t ON d.doc_id = t.doc_id AND t.status = 'ok'
            """
            params: list = []

            if not include_processed:
                base += """
                    LEFT JOIN lemma_results lr
                           ON d.doc_id = lr.doc_id AND lr.method = %s
                    WHERE lr.doc_id IS NULL
                """
                params.append(lemmatizer)

            if limit is not None:
                base += " LIMIT %s"
                params.append(limit)

            with self._conn.cursor() as cur:
                cur.execute(base, params)
                return [_pdf_state_from_db_resp(row) for row in cur.fetchall()]

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    self._conn = None
                except Exception:  # noqa BLE:001
                    pass
