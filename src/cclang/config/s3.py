"""S3/Yandex Object Storage configuration helpers sourced from environment variables."""
from dataclasses import dataclass
from pathlib import Path
import os

from contourpy import max_threads


@dataclass
class S3Config:
    """Runtime S3 credentials and addressing."""
    enable: bool
    bucket: str
    root_prefix: Path
    region: str | None
    access_key: str | None
    secret_key: str | None

def load_s3_config() -> S3Config:
    return S3Config(
        enable=os.environ.get("CCLANG_S3_ENABLE", "false").lower() == "true",
        bucket=os.environ.get("CCLANG_S3_BUCKET", ""),
        root_prefix= Path(os.environ.get("CCLANG_S3_ROOT_PREFIX", "").strip("/")),
        region=os.environ.get("AWS_REGION"),
        access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
