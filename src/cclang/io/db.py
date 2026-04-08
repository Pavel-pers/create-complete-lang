# get connection and ensure schemas for databases
from __future__ import annotations

import os
from typing import Final
import psycopg

DB_DSN_ENV: Final[str] = "CCLANG_DB_DSN"


def get_conn(database_dsn: str | None) -> psycopg.Connection:
    dsn = database_dsn or os.environ.get(DB_DSN_ENV)
    if not dsn:
        raise RuntimeError(f"{DB_DSN_ENV} is not set")

    # Add a short connect timeout so a bad network/host does not hang the pipeline startup.
    conn = psycopg.connect(dsn, connect_timeout=5)
    # Migrations MUST run before ensure_schema to avoid name conflicts
    _migrate_pdf_state(conn)
    _migrate_artifact_tables(conn)
    ensure_schema(conn)
    return conn


def _migrate_pdf_state(conn: psycopg.Connection) -> None:
    """If the legacy ``pdf_state`` table exists, copy data into the new
    normalized tables and rename it to ``pdf_state_old``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'pdf_state'
            )
            """
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return

        # Already migrated in a previous run (pdf_state_old exists)
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'pdf_state_old'
            )
            """
        )
        row = cur.fetchone()
        if row and row[0]:
            # Both tables exist — previous migration copied data but
            # pdf_state was recreated.  Drop the stale original.
            cur.execute("DROP TABLE pdf_state")
            conn.commit()
            return

        # Copy documents
        cur.execute(
            """
            INSERT INTO documents (doc_id, path)
            SELECT pdf_sha, pdf_path FROM pdf_state
            ON CONFLICT (doc_id) DO NOTHING
            """
        )
        # Copy ocr_results (only rows that were actually processed)
        cur.execute(
            """
            INSERT INTO ocr_results (doc_id, artefact_id, status, path, updated_at)
            SELECT pdf_sha, text_sha, text_status, text_path, text_updated_at
            FROM pdf_state
            WHERE text_status IN ('ok', 'error')
            ON CONFLICT (doc_id) DO NOTHING
            """
        )
        # Copy tokenize_results
        cur.execute(
            """
            INSERT INTO tokenize_results (doc_id, artefact_id, status, path, updated_at)
            SELECT pdf_sha, tokenize_sha, tokenize_status, tokenize_path, tokenize_updated_at
            FROM pdf_state
            WHERE tokenize_status IN ('ok', 'error')
            ON CONFLICT (doc_id) DO NOTHING
            """
        )
        # Copy lemma_results (default method for legacy data)
        cur.execute(
            """
            INSERT INTO lemma_results (doc_id, method, artefact_id, status, path, updated_at)
            SELECT pdf_sha, 'apertium-mar-morph', lemma_sha, lemma_status, lemma_path, lemma_updated_at
            FROM pdf_state
            WHERE lemma_status IN ('ok', 'error')
            ON CONFLICT (doc_id, method) DO NOTHING
            """
        )
        # Rename old table
        cur.execute("ALTER TABLE pdf_state RENAME TO pdf_state_old")
    conn.commit()


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fetch_items
            (
                id         BIGSERIAL PRIMARY KEY,
                url        TEXT        NOT NULL,
                sha256     TEXT        NOT NULL,
                local_path TEXT        NOT NULL,
                ts         TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fetch_items_url "
            "ON fetch_items (url);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fetch_items_sha256 "
            "ON fetch_items (sha256);"
        )

        # ---------- normalized tables ----------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents
            (
                doc_id TEXT PRIMARY KEY,
                path   TEXT NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_results
            (
                doc_id      TEXT PRIMARY KEY REFERENCES documents(doc_id),
                artefact_id TEXT,
                status      TEXT NOT NULL DEFAULT 'ok',
                path        TEXT,
                updated_at  TIMESTAMPTZ
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocr_results_status "
            "ON ocr_results (status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocr_results_artefact_id "
            "ON ocr_results (artefact_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tokenize_results
            (
                doc_id      TEXT PRIMARY KEY REFERENCES documents(doc_id),
                artefact_id TEXT,
                status      TEXT NOT NULL DEFAULT 'ok',
                path        TEXT,
                updated_at  TIMESTAMPTZ
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tokenize_results_status "
            "ON tokenize_results (status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tokenize_results_artefact_id "
            "ON tokenize_results (artefact_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lemma_results
            (
                doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
                method      TEXT NOT NULL,
                artefact_id TEXT,
                status      TEXT NOT NULL DEFAULT 'ok',
                path        TEXT,
                updated_at  TIMESTAMPTZ,
                PRIMARY KEY (doc_id, method)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_lemma_results_status "
            "ON lemma_results (status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_lemma_results_artefact_id "
            "ON lemma_results (artefact_id);"
        )

        cur.execute(
            """CREATE TABLE IF NOT EXISTS vocab_builds
               (
                   run_id        SERIAL PRIMARY KEY,
                   lemma_method  TEXT NOT NULL,
                   artefact_id   TEXT,
                   status        TEXT NOT NULL DEFAULT 'ok',
                   path          TEXT,
                   params        JSONB,
                   stats         JSONB,
                   updated_at    TIMESTAMPTZ
               );
            """
        )

        # ---------- fragment pipeline (renamed from corpus_builds) ----------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fragment_builds
            (
                run_id        SERIAL PRIMARY KEY,
                vocab_id      INT  NOT NULL REFERENCES vocab_builds(run_id),
                fragment_size INT  NOT NULL,
                stats         JSONB,
                status        TEXT NOT NULL DEFAULT 'ok',
                updated_at    TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fragment_builds_vocab_id "
            "ON fragment_builds (vocab_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fragment_results
            (
                build_id      INT  NOT NULL REFERENCES fragment_builds (run_id),
                source_doc_id TEXT NOT NULL,
                artefact_id   TEXT,
                status        TEXT NOT NULL DEFAULT 'ok',
                path          TEXT,
                updated_at    TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fragment_results_build_id "
            "ON fragment_results (build_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tdm_builds
            (
                run_id      SERIAL PRIMARY KEY,
                corpus_id   INT NOT NULL REFERENCES fragment_builds(run_id),
                weighting   TEXT,
                stats       JSONB,
                path        TEXT,
                status      TEXT,
                updated_at  TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tdm_corpus_id "
            "ON tdm_builds (corpus_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS svd_builds
            (
                run_id     SERIAL PRIMARY KEY,
                tdm_id     INT NOT NULL REFERENCES tdm_builds(run_id),
                k          INT NOT NULL,
                params     JSONB,
                stats      JSONB,
                path       TEXT,
                status     TEXT NOT NULL DEFAULT 'running',
                updated_at TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_svd_builds_tdm_id "
            "ON svd_builds (tdm_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_builds_old
            (
                run_id      SERIAL PRIMARY KEY,
                svd_id      INT NOT NULL REFERENCES svd_builds(run_id),
                sigma_power DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                reshape_k   INT,
                method      TEXT NOT NULL DEFAULT 'svd',
                stats       JSONB,
                path        TEXT,
                status      TEXT NOT NULL DEFAULT 'running',
                updated_at  TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_builds_old_svd_id "
            "ON embedding_builds_old (svd_id);"
        )

        # ---------- unified artifact tables ----------

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_builds
            (
                run_id     SERIAL PRIMARY KEY,
                path       TEXT NOT NULL,
                version    TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'running',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_builds
            (
                run_id     SERIAL PRIMARY KEY,
                path       TEXT NOT NULL,
                version    TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'running',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_builds
            (
                run_id     SERIAL PRIMARY KEY,
                path       TEXT NOT NULL,
                version    TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'running',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    conn.commit()


def _migrate_artifact_tables(conn: psycopg.Connection) -> None:
    """Rename old ``corpus_builds`` → ``fragment_builds``,
    ``corpus_fragments`` → ``fragment_results``,
    ``embedding_builds`` → ``embedding_builds_old``.
    Then the new unified ``corpus_builds``, ``embedding_builds``,
    ``cluster_builds`` are created by ``ensure_schema()``."""
    with conn.cursor() as cur:
        # Check if migration already done (fragment_builds exists)
        cur.execute("SELECT to_regclass('public.fragment_builds')")
        if cur.fetchone()[0] is not None:
            return

        # Check if old corpus_builds exists (nothing to migrate on fresh DB)
        cur.execute("SELECT to_regclass('public.corpus_builds')")
        row = cur.fetchone()
        old_corpus_exists = row[0] is not None

        if old_corpus_exists:
            # Check if it's the OLD schema (has vocab_id column) or new one
            cur.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'corpus_builds' AND column_name = 'vocab_id'
                """
            )
            is_old_schema = cur.fetchone() is not None

            if not is_old_schema:
                # Already the new unified schema, nothing to migrate
                return

            # Rename old tables
            cur.execute("ALTER TABLE corpus_builds RENAME TO fragment_builds")
            cur.execute("ALTER TABLE corpus_fragments RENAME TO fragment_results")
            cur.execute(
                "ALTER INDEX IF EXISTS idx_corpus_builds_vocab_id "
                "RENAME TO idx_fragment_builds_vocab_id"
            )
            cur.execute(
                "ALTER INDEX IF EXISTS idx_corpus_fragments_build_id "
                "RENAME TO idx_fragment_results_build_id"
            )

        # Check if old embedding_builds exists (has svd_id column)
        cur.execute("SELECT to_regclass('public.embedding_builds')")
        row = cur.fetchone()
        old_emb_exists = row[0] is not None

        if old_emb_exists:
            cur.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'embedding_builds' AND column_name = 'svd_id'
                """
            )
            is_old_emb_schema = cur.fetchone() is not None

            if is_old_emb_schema:
                cur.execute(
                    "ALTER TABLE embedding_builds RENAME TO embedding_builds_old"
                )
                cur.execute(
                    "ALTER INDEX IF EXISTS idx_embedding_builds_svd_id "
                    "RENAME TO idx_embedding_builds_old_svd_id"
                )

        # Create the new unified tables (idempotent)
        for table in ("corpus_builds", "embedding_builds", "cluster_builds"):
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table}
                (
                    run_id     SERIAL PRIMARY KEY,
                    path       TEXT NOT NULL,
                    version    TEXT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'running',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    conn.commit()


"""
sample of sqlite3 migration:
pgloader \
  sqlite:///data/databases/pipelines.sqlite3 \
  postgresql://cclang:<pswrd>@<ip_rpi>:5432/cclang_db
"""
