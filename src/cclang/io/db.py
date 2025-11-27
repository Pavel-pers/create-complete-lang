# get connection and ensure schemas for databases
import sqlite3
from pathlib import Path


def get_conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS fetch_items
        (
            url        TEXT NOT NULL,
            sha256     TEXT NOT NULL,
            local_path TEXT NOT NULL,
            ts         TEXT NOT NULL

        );
        CREATE INDEX IF NOT EXISTS idx_fetch_items_url ON fetch_items (url);
        CREATE INDEX IF NOT EXISTS idx_fetch_items_sha256 ON fetch_items (sha256);
        CREATE TABLE IF NOT EXISTS pdf_state
        (
            pdf_sha          TEXT PRIMARY KEY,
            pdf_path         TEXT NOT NULL,

            text_sha         TEXT,
            text_path        TEXT,
            text_status      TEXT,
            text_updated_at  TEXT,

            lemma_sha        TEXT,
            lemma_path       TEXT,
            lemma_status     TEXT,
            lemma_updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pdf_state_text_sha ON pdf_state (text_sha);
        CREATE INDEX IF NOT EXISTS idx_pdf_state_text_status ON pdf_state (text_status);  
        CREATE INDEX IF NOT EXISTS idx_pdf_state_lemma_sha ON pdf_state (lemma_sha);
        CREATE INDEX IF NOT EXISTS idx_pdf_state_lemma_status ON pdf_state (lemma_status);
        """)
    conn.commit()
