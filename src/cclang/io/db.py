import sqlite3
from pathlib import Path

def get_conn(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    ensure_schema(conn)
    return conn

def ensure_schema(conn: sqlite3.Connection)->None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fetch_items (
            url TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            local_path TEXT NOT NULL,
            ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fetch_items_url ON fetch_items (url);
        CREATE INDEX IF NOT EXISTS idx_fetch_items_sha256 ON fetch_items (sha256);
    """)
    conn.commit()