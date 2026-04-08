from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import psycopg

from cclang.io.schemas import (
    ArtifactBuildInfo,
    FragmentBuildInfo,
    LemmaResult,
    LemmatizeTask,
    OcrResult,
    OcrTask,
    ProcessingStatus,
    SvdBuildInfo,
    TdmInfo,
    TokenizeTask,
    VocabInfo,
    VocabParams,
    VocabStats,
    EmbeddingsBuildInfo,
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

    # * documents

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

    # * ocr_results

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
                            WHERE o.doc_id IS NULL \
                            """
                    params: list = []
                else:
                    query = """
                            SELECT d.doc_id, d.path
                            FROM documents d
                                     JOIN ocr_results o ON d.doc_id = o.doc_id
                            WHERE o.status = %s \
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
                        SET status      = EXCLUDED.status,
                            artefact_id = EXCLUDED.artefact_id,
                            path        = EXCLUDED.path,
                            updated_at  = EXCLUDED.updated_at
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

    # * tokenize_results

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
                            WHERE t.doc_id IS NULL \
                            """
                    params: list = []
                else:
                    query = """
                            SELECT d.doc_id, o.path AS text_path
                            FROM documents d
                                     JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                                     JOIN tokenize_results t ON d.doc_id = t.doc_id
                            WHERE t.status = %s \
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
                        SET status      = EXCLUDED.status,
                            artefact_id = EXCLUDED.artefact_id,
                            path        = EXCLUDED.path,
                            updated_at  = EXCLUDED.updated_at
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

    # * lemma_results

    def get_lemma_tasks(
            self,
            target_status: Optional[ProcessingStatus] = ProcessingStatus.NOT_PROCESSED,
            method: str = "apertium-mar-morph",
            limit: Optional[int] = None,
    ) -> List[LemmatizeTask]:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                if target_status is None:
                    # all — every tokenized doc regardless of lemma status
                    query = """
                            SELECT d.doc_id, t.path AS tokenize_path
                            FROM documents d
                                     JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                                     JOIN tokenize_results t ON d.doc_id = t.doc_id AND t.status = 'ok' \
                            """
                    params: list = []
                elif target_status == ProcessingStatus.NOT_PROCESSED:
                    # pending — docs with no lemma result for this method
                    query = """
                            SELECT d.doc_id, t.path AS tokenize_path
                            FROM documents d
                                     JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                                     JOIN tokenize_results t ON d.doc_id = t.doc_id AND t.status = 'ok'
                                     LEFT JOIN lemma_results l
                                               ON d.doc_id = l.doc_id AND l.method = %s
                            WHERE l.doc_id IS NULL \
                            """
                    params = [method]
                else:
                    query = """
                            SELECT d.doc_id, t.path AS tokenize_path
                            FROM documents d
                                     JOIN ocr_results o ON d.doc_id = o.doc_id AND o.status = 'ok'
                                     JOIN tokenize_results t ON d.doc_id = t.doc_id AND t.status = 'ok'
                                     JOIN lemma_results l
                                          ON d.doc_id = l.doc_id AND l.method = %s
                            WHERE l.status = %s \
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
                        SET status      = EXCLUDED.status,
                            artefact_id = EXCLUDED.artefact_id,
                            path        = EXCLUDED.path,
                            updated_at  = EXCLUDED.updated_at
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
                        WHERE status = %s
                          AND method = %s
                        ORDER BY doc_id
                        """,
                        (ProcessingStatus.OK.value, method),
                    )
                else:
                    cur.execute(
                        """
                        SELECT doc_id, method, artefact_id, status, path, updated_at
                        FROM lemma_results
                        WHERE status = %s
                        ORDER BY doc_id
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

    # * vocab_builds

    def insert_vocab_build(
            self,
            lemma_method: str,
            artefact_id: Optional[str],
            status: ProcessingStatus,
            path: Optional[str],
            params: Optional[VocabParams] = None,
            stats: Optional[VocabStats] = None,
    ) -> int:
        """Insert a vocab build record, return run_id."""
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vocab_builds
                        (lemma_method, artefact_id, status, path, params, stats, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
                    RETURNING run_id
                    """,
                    (
                        lemma_method,
                        artefact_id,
                        status.value,
                        path,
                        params.model_dump_json() if params else None,
                        stats.model_dump_json() if stats else None,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                run_id = row[0]
            conn.commit()
            return run_id

    # * corpus_builds / corpus_fragments

    def get_vocab_build(self, run_id: int) -> VocabInfo:
        """Return a VocabInfo for the given vocab_builds.run_id, or raise ValueError."""
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id,
                           lemma_method,
                           artefact_id,
                           status,
                           path,
                           params,
                           stats,
                           updated_at
                    FROM vocab_builds
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"vocab_builds run_id={run_id} not found")
                return VocabInfo(
                    run_id=row[0],
                    lemma_method=row[1],
                    artefact_id=row[2],
                    status=ProcessingStatus(row[3]),
                    path=row[4],
                    vocab_params=VocabParams.model_validate(row[5]) if row[5] else None,
                    vocab_stats=VocabStats.model_validate(row[6]) if row[6] else None,
                    updated_at=(
                        row[7].isoformat() if isinstance(row[7], datetime) else row[7]
                    ),
                )

    def get_fragment_build(self, run_id: int) -> FragmentBuildInfo:
        """Return a FragmentBuildInfo for the given fragment_builds.run_id, or raise ValueError."""
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id,
                           vocab_id,
                           fragment_size,
                           status,
                           stats,
                           updated_at
                    FROM fragment_builds
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"fragment_builds run_id={run_id} not found")
                return FragmentBuildInfo(
                    run_id=row[0],
                    vocab_id=row[1],
                    fragment_size=row[2],
                    status=ProcessingStatus(row[3]),
                    stats=row[4],
                    updated_at=(
                        row[5].isoformat() if isinstance(row[5], datetime) else row[5]
                    ),
                )

    def get_tdm_build(self, run_id: int) -> TdmInfo:
        """Return a TdmInfo for the given tdm_builds.run_id, or raise ValueError."""
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id,
                           corpus_id,
                           weighting,
                           status,
                           path,
                           stats,
                           updated_at
                    FROM tdm_builds
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"tdm_builds run_id={run_id} not found")
                return TdmInfo(
                    run_id=row[0],
                    corpus_id=row[1],
                    weighting=row[2],
                    status=ProcessingStatus(row[3]),
                    path=row[4],
                    stats=row[5],
                    updated_at=(
                        row[6].isoformat() if isinstance(row[6], datetime) else row[6]
                    ),
                )

    def get_fragments_by_build_id(self, build_id: int) -> list[str]:
        """
            returns fragments path by build_id
        """
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT path
                    FROM fragment_results
                    WHERE status = 'ok' AND build_id = %s
                    """, (build_id, ), )
                rows = cur.fetchall()
                return [row[0] for row in rows]

    def insert_fragment_build(
            self,
            vocab_id: int,
            fragment_size: int,
            status: ProcessingStatus,
            stats: dict[str, int] | None = None,
    ) -> int:
        """Insert a fragment_builds record and return run_id."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO fragment_builds (vocab_id, fragment_size, stats, status, updated_at)
                    VALUES (%s, %s, %s::jsonb, %s, NOW())
                    RETURNING run_id
                    """,
                    (
                        vocab_id,
                        fragment_size,
                        json.dumps(stats) if stats else None,
                        status.value,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                run_id: int = row[0]
            conn.commit()
            return run_id

    def update_fragment_build(
            self,
            build_id: int,
            status: ProcessingStatus,
            stats: dict[str, int] | None = None,
    ) -> None:
        """Update status and stats for a fragment_builds record."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE fragment_builds
                    SET status     = %s,
                        stats      = %s::jsonb,
                        updated_at = NOW()
                    WHERE run_id = %s
                    """,
                    (
                        status.value,
                        json.dumps(stats) if stats else None,
                        build_id,
                    ),
                )
            conn.commit()

    def insert_tdm_build(
            self,
            corpus_id: int,
            weighting: str,
            status: ProcessingStatus,
            path: str | None = None,
            stats: dict[str, int] | None = None,
    ) -> int:
        """Insert a tdm_builds record, return run_id."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tdm_builds
                        (corpus_id, weighting, status, path, stats, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    RETURNING run_id
                    """,
                    (
                        corpus_id,
                        weighting,
                        status.value,
                        path,
                        json.dumps(stats) if stats else None,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                run_id: int = row[0]
            conn.commit()
            return run_id

    def update_tdm_build(
            self,
            build_id: int,
            status: ProcessingStatus,
            path: str | None = None,
            stats: dict[str, int] | None = None,
    ) -> None:
        """Update status, path, and stats for a tdm_builds record."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tdm_builds
                    SET status     = %s,
                        path       = %s,
                        stats      = %s::jsonb,
                        updated_at = NOW()
                    WHERE run_id = %s
                    """,
                    (
                        status.value,
                        path,
                        json.dumps(stats) if stats else None,
                        build_id,
                    ),
                )
            conn.commit()

    # * svd_builds

    def insert_svd_build(
            self,
            tdm_id: int,
            k: int,
            status: ProcessingStatus,
            params: dict[str, Any] | None = None,
            path: str | None = None,
            stats: dict[str, Any] | None = None,
    ) -> int:
        """Insert an svd_builds record and return run_id."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO svd_builds
                        (tdm_id, k, params, stats, path, status, updated_at)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, NOW())
                    RETURNING run_id
                    """,
                    (
                        tdm_id,
                        k,
                        json.dumps(params) if params else None,
                        json.dumps(stats) if stats else None,
                        path,
                        status.value,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                run_id: int = row[0]
            conn.commit()
            return run_id

    def update_svd_build(
            self,
            build_id: int,
            status: ProcessingStatus,
            path: str | None = None,
            stats: dict[str, Any] | None = None,
    ) -> None:
        """Update status, path, and stats for an svd_builds record."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE svd_builds
                    SET status     = %s,
                        path       = %s,
                        stats      = %s::jsonb,
                        updated_at = NOW()
                    WHERE run_id = %s
                    """,
                    (
                        status.value,
                        path,
                        json.dumps(stats) if stats else None,
                        build_id,
                    ),
                )
            conn.commit()

    def get_svd_build(self, run_id: int) -> SvdBuildInfo:
        """Return a SvdBuildInfo for the given svd_builds.run_id, or raise ValueError."""
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id,
                           tdm_id,
                           k,
                           params,
                           stats,
                           path,
                           status,
                           updated_at
                    FROM svd_builds
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"svd_builds run_id={run_id} not found")
                return SvdBuildInfo(
                    run_id=row[0],
                    tdm_id=row[1],
                    k=row[2],
                    params=row[3],
                    stats=row[4],
                    path=row[5],
                    status=ProcessingStatus(row[6]),
                    updated_at=(
                        row[7].isoformat() if isinstance(row[7], datetime) else row[7]
                    ),
                )

    # * embedding_builds_old (SVD pipeline legacy)

    def insert_svd_embedding_build(
            self,
            svd_id: int,
            sigma_power: float,
            method: str,
            status: ProcessingStatus,
            reshape_k: int | None = None,
            path: str | None = None,
            stats: dict[str, Any] | None = None,
    ) -> int:
        """Insert an embedding_builds_old record and return run_id."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO embedding_builds_old
                        (svd_id, sigma_power, reshape_k, method, stats, path, status, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, NOW())
                    RETURNING run_id
                    """,
                    (
                        svd_id,
                        sigma_power,
                        reshape_k,
                        method,
                        json.dumps(stats) if stats else None,
                        path,
                        status.value,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                run_id: int = row[0]
            conn.commit()
            return run_id

    def update_svd_embedding_build(
            self,
            build_id: int,
            status: ProcessingStatus,
            path: str | None = None,
            stats: dict[str, Any] | None = None,
    ) -> None:
        """Update status, path, and stats for an embedding_builds_old record."""
        import json

        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE embedding_builds_old
                    SET status     = %s,
                        path       = %s,
                        stats      = %s::jsonb,
                        updated_at = NOW()
                    WHERE run_id = %s
                    """,
                    (
                        status.value,
                        path,
                        json.dumps(stats) if stats else None,
                        build_id,
                    ),
                )
            conn.commit()

    def get_svd_embedding_build(self, run_id: int) -> EmbeddingsBuildInfo:
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id,
                            svd_id,
                            sigma_power,
                            reshape_k,
                            method,
                            stats,
                            path,
                            status,
                            updated_at
                    FROM embedding_builds_old
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"embedding_builds_old run_id={run_id} not found")
                return EmbeddingsBuildInfo(
                    run_id=row[0],
                    svd_id=row[1],
                    sigma_power=row[2],
                    reshape_k=row[3],
                    method=row[4],
                    stats=row[5],
                    path=row[6],
                    status=ProcessingStatus(row[7]),
                    updated_at=(
                        row[8].isoformat() if isinstance(row[8], datetime) else row[8]
                    ),
                )


    def batch_insert_fragment_results(
            self,
            rows: list[tuple[int, str, str | None, str, str | None]],
    ) -> None:
        """Batch INSERT rows into fragment_results via executemany.

        Each row = (build_id, source_doc_id, artefact_id, status, path).
        """
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO fragment_results
                        (build_id, source_doc_id, artefact_id, status, path, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    rows,
                )
            conn.commit()

    # ────────────────────────────────────────────────────────
    # Unified artifact builds (corpus, embedding, cluster)
    # ────────────────────────────────────────────────────────

    _ARTIFACT_TABLES = frozenset({"corpus_builds", "embedding_builds", "cluster_builds"})

    def _register_build(
            self, table: str, path: str, version: str,
            status: ProcessingStatus = ProcessingStatus.RUNNING,
    ) -> int:
        assert table in self._ARTIFACT_TABLES
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {table} (path, version, status, created_at, updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    RETURNING run_id
                    """,
                    (path, version, status.value),
                )
                row = cur.fetchone()
                assert row is not None
                run_id: int = row[0]
            conn.commit()
            return run_id

    def _complete_build(
            self, table: str, run_id: int,
            status: ProcessingStatus,
    ) -> None:
        assert table in self._ARTIFACT_TABLES
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {table}
                    SET status = %s, updated_at = NOW()
                    WHERE run_id = %s
                    """,
                    (status.value, run_id),
                )
            conn.commit()

    def _get_build(self, table: str, run_id: int) -> ArtifactBuildInfo:
        assert table in self._ARTIFACT_TABLES
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT run_id, path, version, status, created_at, updated_at
                    FROM {table}
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"{table} run_id={run_id} not found")
                return ArtifactBuildInfo(
                    run_id=row[0],
                    path=row[1],
                    version=row[2],
                    status=ProcessingStatus(row[3]),
                    created_at=(
                        row[4].isoformat() if isinstance(row[4], datetime) else row[4]
                    ),
                    updated_at=(
                        row[5].isoformat() if isinstance(row[5], datetime) else row[5]
                    ),
                )

    def _list_builds(
            self, table: str,
            status: ProcessingStatus | None = None,
    ) -> list[ArtifactBuildInfo]:
        assert table in self._ARTIFACT_TABLES
        with self._lock:
            conn = self._check_conn()
            with conn.cursor() as cur:
                if status is not None:
                    cur.execute(
                        f"""
                        SELECT run_id, path, version, status, created_at, updated_at
                        FROM {table}
                        WHERE status = %s
                        ORDER BY run_id
                        """,
                        (status.value,),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT run_id, path, version, status, created_at, updated_at
                        FROM {table}
                        ORDER BY run_id
                        """
                    )
                results: list[ArtifactBuildInfo] = []
                for row in cur.fetchall():
                    results.append(ArtifactBuildInfo(
                        run_id=row[0],
                        path=row[1],
                        version=row[2],
                        status=ProcessingStatus(row[3]),
                        created_at=(
                            row[4].isoformat() if isinstance(row[4], datetime) else row[4]
                        ),
                        updated_at=(
                            row[5].isoformat() if isinstance(row[5], datetime) else row[5]
                        ),
                    ))
                return results

    # * corpus_builds (unified)

    def register_corpus_build(
            self, path: str, version: str,
            status: ProcessingStatus = ProcessingStatus.RUNNING,
    ) -> int:
        return self._register_build("corpus_builds", path, version, status)

    def complete_corpus_build(self, run_id: int, status: ProcessingStatus) -> None:
        self._complete_build("corpus_builds", run_id, status)

    def get_corpus_build(self, run_id: int) -> ArtifactBuildInfo:
        return self._get_build("corpus_builds", run_id)

    def list_corpus_builds(
            self, status: ProcessingStatus | None = None,
    ) -> list[ArtifactBuildInfo]:
        return self._list_builds("corpus_builds", status)

    # * embedding_builds (unified)

    def register_embedding_build(
            self, path: str, version: str,
            status: ProcessingStatus = ProcessingStatus.RUNNING,
    ) -> int:
        return self._register_build("embedding_builds", path, version, status)

    def complete_embedding_build(self, run_id: int, status: ProcessingStatus) -> None:
        self._complete_build("embedding_builds", run_id, status)

    def get_embedding_build(self, run_id: int) -> ArtifactBuildInfo:
        return self._get_build("embedding_builds", run_id)

    def list_embedding_builds(
            self, status: ProcessingStatus | None = None,
    ) -> list[ArtifactBuildInfo]:
        return self._list_builds("embedding_builds", status)

    # * cluster_builds (unified)

    def register_cluster_build(
            self, path: str, version: str,
            status: ProcessingStatus = ProcessingStatus.RUNNING,
    ) -> int:
        return self._register_build("cluster_builds", path, version, status)

    def complete_cluster_build(self, run_id: int, status: ProcessingStatus) -> None:
        self._complete_build("cluster_builds", run_id, status)

    def get_cluster_build(self, run_id: int) -> ArtifactBuildInfo:
        return self._get_build("cluster_builds", run_id)

    def list_cluster_builds(
            self, status: ProcessingStatus | None = None,
    ) -> list[ArtifactBuildInfo]:
        return self._list_builds("cluster_builds", status)

    # lifecycle

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._conn = None
