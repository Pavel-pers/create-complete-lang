from __future__ import annotations

import threading
from datetime import datetime
from typing import List, Optional

import psycopg

from cclang.io.schemas import (
    LemmaResult,
    LemmatizeTask,
    OcrResult,
    OcrTask,
    ProcessingStatus,
    TokenizeTask,
)


class DocStateStore:
    """
    Queries the ``documents``, ``ocr_results``, ``tokenize_results`` and
    ``lemma_results`` tables instead of the monolithic ``pdf_state`` table.
    All public methods are thread-safe (guarded by a lock).
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn: Optional[psycopg.Connection] = conn
        self._lock = threading.Lock()

    def _check_conn(self) -> psycopg.Connection:
        if self._conn is None:
            raise RuntimeError("invalid access to closed connection")
        return self._conn

    #* documents

    def sync_documents_from_fetch_items(self) -> None:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, path)
                    SELECT fi.sha256     AS doc_id,
                           fi.local_path AS path
                    FROM fetch_items AS fi
                    ON CONFLICT (doc_id) DO NOTHING;
                    """
                )
            conn.commit()

    def get_all_doc_paths(self) -> List[str]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT path FROM documents")
                return [row[0] for row in cur.fetchall()]

    #* ocr_results

    def get_ocr_tasks(
        self,
        target_status: Optional[ProcessingStatus] = ProcessingStatus.NOT_PROCESSED,
        limit: Optional[int] = None,
    ) -> List[OcrTask]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                if target_status is None or target_status == ProcessingStatus.NOT_PROCESSED:
                    # pending = no row in ocr_results
                    query = """
                        SELECT d.doc_id, d.path
                        FROM documents d
                        LEFT JOIN ocr_results o ON d.doc_id = o.doc_id
                        WHERE o.doc_id IS NULL
                    """
                    params: list = []
                else:
                    query = """
                        SELECT d.doc_id, d.path
                        FROM documents d
                        JOIN ocr_results o ON d.doc_id = o.doc_id
                        WHERE o.status = %s
                    """
                    params = [target_status.value]

                if limit is not None:
                    query += " LIMIT %s"
                    params.append(limit)

                cur.execute(query, params)
                return [OcrTask(doc_id=row[0], pdf_path=row[1]) for row in cur.fetchall()]

    def get_ocr_result_by_artefact(self, artefact_id: str) -> Optional[OcrResult]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT doc_id, artefact_id, status, path, updated_at
                    FROM ocr_results
                    WHERE artefact_id = %s
                    """,
                    (artefact_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return OcrResult(
                    doc_id=row[0],
                    artefact_id=row[1],
                    status=ProcessingStatus(row[2]),
                    path=row[3],
                    updated_at=row[4].isoformat() if isinstance(row[4], datetime) else row[4],
                )

    def upsert_ocr_result(
        self,
        doc_id: str,
        status: ProcessingStatus,
        artefact_id: Optional[str],
        path: Optional[str],
        ts: Optional[str],
    ) -> None:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ocr_results (doc_id, status, artefact_id, path, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                    SET status     = EXCLUDED.status,
                        artefact_id = EXCLUDED.artefact_id,
                        path       = EXCLUDED.path,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (doc_id, status.value, artefact_id, path, ts),
                )
            conn.commit()

    def get_completed_ocr_paths(self) -> List[str]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT path FROM ocr_results WHERE status = %s AND path IS NOT NULL",
                    (ProcessingStatus.OK.value,),
                )
                return [row[0] for row in cur.fetchall()]

    #* tokenize_results

    def get_tokenize_tasks(
        self,
        target_status: Optional[ProcessingStatus] = ProcessingStatus.NOT_PROCESSED,
        limit: Optional[int] = None,
    ) -> List[TokenizeTask]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                if target_status is None or target_status == ProcessingStatus.NOT_PROCESSED:
                    query = """
                        SELECT d.doc_id, o.path AS text_path
                        FROM documents d
                        JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                        LEFT JOIN tokenize_results t ON d.doc_id = t.doc_id
                        WHERE t.doc_id IS NULL
                    """
                    params: list = []
                else:
                    query = """
                        SELECT d.doc_id, o.path AS text_path
                        FROM documents d
                        JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                        JOIN tokenize_results t ON d.doc_id = t.doc_id
                        WHERE t.status = %s
                    """
                    params = [target_status.value]

                if limit is not None:
                    query += " LIMIT %s"
                    params.append(limit)

                cur.execute(query, params)
                return [TokenizeTask(doc_id=row[0], text_path=row[1]) for row in cur.fetchall()]

    def upsert_tokenize_result(
        self,
        doc_id: str,
        status: ProcessingStatus,
        artefact_id: Optional[str],
        path: Optional[str],
        ts: Optional[str],
    ) -> None:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tokenize_results (doc_id, status, artefact_id, path, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                    SET status     = EXCLUDED.status,
                        artefact_id = EXCLUDED.artefact_id,
                        path       = EXCLUDED.path,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (doc_id, status.value, artefact_id, path, ts),
                )
            conn.commit()

    def get_completed_tokenize_paths(self) -> List[str]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT path FROM tokenize_results WHERE status = %s AND path IS NOT NULL",
                    (ProcessingStatus.OK.value,),
                )
                return [row[0] for row in cur.fetchall()]

    #* lemma_results

    def get_lemma_tasks(
        self,
        target_status: Optional[ProcessingStatus] = ProcessingStatus.NOT_PROCESSED,
        method: str = "apertium-mar-morph",
        limit: Optional[int] = None,
    ) -> List[LemmatizeTask]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                if target_status is None or target_status == ProcessingStatus.NOT_PROCESSED:
                    query = """
                        SELECT d.doc_id, t.path AS tokenize_path
                        FROM documents d
                        JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                        JOIN tokenize_results t ON d.doc_id = t.doc_id AND t.status = 'ok'
                        LEFT JOIN lemma_results l
                            ON d.doc_id = l.doc_id AND l.method = %s
                        WHERE l.doc_id IS NULL
                    """
                    params: list = [method]
                else:
                    query = """
                        SELECT d.doc_id, t.path AS tokenize_path
                        FROM documents d
                        JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                        JOIN tokenize_results t ON d.doc_id = t.doc_id AND t.status = 'ok'
                        JOIN lemma_results l
                            ON d.doc_id = l.doc_id AND l.method = %s
                        WHERE l.status = %s
                    """
                    params = [method, target_status.value]

                if limit is not None:
                    query += " LIMIT %s"
                    params.append(limit)

                cur.execute(query, params)
                return [
                    LemmatizeTask(doc_id=row[0], tokenize_path=row[1])
                    for row in cur.fetchall()
                ]

    def upsert_lemma_result(
        self,
        doc_id: str,
        method: str,
        status: ProcessingStatus,
        artefact_id: Optional[str],
        path: Optional[str],
        ts: Optional[str],
    ) -> None:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lemma_results (doc_id, method, status, artefact_id, path, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, method) DO UPDATE
                    SET status     = EXCLUDED.status,
                        artefact_id = EXCLUDED.artefact_id,
                        path       = EXCLUDED.path,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (doc_id, method, status.value, artefact_id, path, ts),
                )
            conn.commit()

    def get_completed_lemma_results(
        self, method: Optional[str] = None
    ) -> List[LemmaResult]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                if method is not None:
                    cur.execute(
                        """
                        SELECT doc_id, method, artefact_id, status, path, updated_at
                        FROM lemma_results
                        WHERE status = %s AND method = %s
                        """,
                        (ProcessingStatus.OK.value, method),
                    )
                else:
                    cur.execute(
                        """
                        SELECT doc_id, method, artefact_id, status, path, updated_at
                        FROM lemma_results
                        WHERE status = %s
                        """,
                        (ProcessingStatus.OK.value,),
                    )
                results: List[LemmaResult] = []
                for row in cur.fetchall():
                    results.append(
                        LemmaResult(
                            doc_id=row[0],
                            method=row[1],
                            artefact_id=row[2],
                            status=ProcessingStatus(row[3]),
                            path=row[4],
                            updated_at=(
                                row[5].isoformat()
                                if isinstance(row[5], datetime)
                                else row[5]
                            ),
                        )
                    )
                return results

    # lifecycle

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._conn = None
