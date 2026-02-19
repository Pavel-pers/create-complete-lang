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

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_builds
            (
                run_id            SERIAL PRIMARY KEY,
                vocab_id      INT  NOT NULL REFERENCES vocab_builds(run_id),
                fragment_size INT  NOT NULL,
                stats         JSONB,
                status        TEXT NOT NULL DEFAULT 'ok',
                updated_at    TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_builds_vocab_id "
            "ON corpus_builds (vocab_id);"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_fragments
            (
                build_id      INT  NOT NULL REFERENCES corpus_builds (run_id),
                source_doc_id TEXT NOT NULL,
                artefact_id   TEXT,
                status        TEXT NOT NULL DEFAULT 'ok',
                path          TEXT,
                updated_at    TIMESTAMPTZ
            )
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_fragments_build_id "
            "ON corpus_fragments (build_id);"
        )

    conn.commit()

    # Migrate legacy data if pdf_state still exists
    _migrate_pdf_state(conn)


"""
sample of sqlite3 migration:
pgloader \
  sqlite:///data/databases/pipelines.sqlite3 \
  postgresql://cclang:<pswrd>@<ip_rpi>:5432/cclang_db
"""
