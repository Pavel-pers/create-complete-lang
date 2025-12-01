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
    conn.commit()


"""
sample of sqlite3 migration:
pgloader \
  sqlite:///data/databases/pipelines.sqlite3 \
  postgresql://cclang:<pswrd>@<ip_rpi>:5432/cclang_db
"""
