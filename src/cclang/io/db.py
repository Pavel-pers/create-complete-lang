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

    conn = psycopg.connect(dsn)
    ensure_schema(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """
    """
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
        cur.execute(
           """
                   CREATE TABLE IF NOT EXISTS pdf_state
        (
            pdf_sha             TEXT PRIMARY KEY,
            pdf_path            TEXT NOT NULL,

            text_sha            TEXT,
            text_path           TEXT,
            text_status         TEXT,
            text_updated_at     TIMESTAMPTZ,

            tokenize_sha        TEXT,
            tokenize_path       TEXT,
            tokenize_status     TEXT,
            tokenize_updated_at TIMESTAMPTZ,

            lemma_sha           TEXT,
            lemma_path          TEXT,
            lemma_status        TEXT,
            lemma_updated_at    TIMESTAMPTZ
        );
           """
        )
        cur.execute(
          "CREATE INDEX IF NOT EXISTS idx_pdf_state_text_sha "
          "ON pdf_state (text_sha);"
        )
        cur.execute(
          "CREATE INDEX IF NOT EXISTS idx_pdf_state_text_status "
          "ON pdf_state (text_status);"
        )
        cur.execute(
          "CREATE INDEX IF NOT EXISTS idx_pdf_state_tokenize_sha "
          "ON pdf_state (tokenize_sha);"
        )
        cur.execute(
          "CREATE INDEX IF NOT EXISTS idx_pdf_state_tokenize_status "
          "ON pdf_state (tokenize_status);"
        )
        cur.execute(
          "CREATE INDEX IF NOT EXISTS idx_pdf_state_lemma_sha "
          "ON pdf_state (lemma_sha);"
        )
        cur.execute(
          "CREATE INDEX IF NOT EXISTS idx_pdf_state_lemma_status "
          "ON pdf_state (lemma_status);"
        )
        
    conn.commit()


"""
sample of sqlite3 migration:
pgloader \
  sqlite:///data/databases/pipelines.sqlite3 \
  postgresql://cclang:<pswrd>@<ip_rpi>:5432/cclang_db
"""
