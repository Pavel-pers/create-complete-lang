"""
Shared setup for StorageManager, DocStateStore, and env loading.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from cclang.config.s3 import load_s3_config
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.fs import LocalConfig

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DATA = EXPERIMENT_ROOT / "data"


def init_env() -> None:
    """Load .env from project root."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def make_storage(save_local: bool = True, cache_files: bool = True) -> StorageManager:
    """Create StorageManager with S3 enabled for reading corpus.

    Matches the convention from existing pipelines:
    - LocalConfig.base_path = data/  (resolved to absolute)
    - CloudConfig.base_path = data/  (appended to S3 root_prefix → cclang/data/)
    So relative keys like 'artifacts/corpus/...' resolve correctly both locally and on S3.
    """
    init_env()
    s3_cfg = load_s3_config()

    local_cfg = LocalConfig(
        base_path=DATA_ROOT,
        save_local=save_local,
        cache_files=cache_files,
        temp_base=DATA_ROOT / "temp" / "wishart_exp",
    )
    cloud_cfg = CloudConfig(
        enable=s3_cfg.enable,
        s3_config=s3_cfg,
        base_path=Path("data"),
        max_upload_threads=0,
    )
    return StorageManager(local_cfg=local_cfg, cloud_cfg=cloud_cfg)


def make_doc_store() -> DocStateStore:
    """Create DocStateStore from env DB DSN."""
    import os
    init_env()
    dsn = os.environ.get("CCLANG_DB_DSN", "")
    if not dsn:
        raise RuntimeError("CCLANG_DB_DSN not set in .env")
    conn = get_conn(dsn)
    return DocStateStore(conn)
