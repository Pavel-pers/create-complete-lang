"""CLI to upload local data directories to S3/Yandex Object Storage with optional concurrency."""

from __future__ import annotations

import argparse
import os
import signal
import threading
from pathlib import Path
from typing import Iterable, Sequence
import json

from cclang.common import logx
from cclang.config.s3 import S3Config, load_s3_config
from cclang.ingest.fs import LocalConfig, CloudConfig, FileManager
from cclang.io.cloud import UploadCallBack

class LoggingCallBack(UploadCallBack):
    def __init__(self, key: str, logger: logx.BoundLogger):
        self.logger = logger
        self.key = key

    def on_upload_succes(self):
        self.logger.info(f"Key {self.key} uploaded successfully")

    def on_upload_failed(self, exc_type, exc_value, traceback):
        self.logger.exception(f"Failed to upload {self.key}", exc_info=(exc_type, exc_value, traceback))


logger = logx.get_logger(__name__)


DEFAULT_SUBDIRS: tuple[str, ...] = ("raw_pdfs", "corpora", "artifacts")


def _resolve_data_path(cli_value: str | None) -> Path:
    """
    Resolve the local data root:
    1) --data-path if provided
    2) env CCLANG_DATA_DIR if set
    3) fail fast if neither is set
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    env_val = os.environ.get("CCLANG_DATA_DIR")
    if env_val:
        return Path(env_val).expanduser().resolve()

    raise SystemExit("CCLANG_DATA_DIR is not set")


def _build_s3_config_from_env_and_args(args: argparse.Namespace) -> S3Config:
    """Merge CLI args with environment to build S3Config, fail fast on missing required fields."""
    config = load_s3_config()
    logger.info("S3 config: %s", config)

    config.bucket = args.bucket or config.bucket
    if not config.bucket:
        raise SystemExit(
            "S3 bucket is not set. Use --bucket or env CCLANG_S3_BUCKET"
        )

    config.region = args.region or config.region
    if not config.region:
        raise SystemExit(
            "S3 region is not set. Use --region or env AWS_REGION"
        )


    config.access_key = args.access_key or config.access_key
    config.secret_key = args.secret_key or config.secret_key
    if not config.access_key or not config.secret_key:
        raise SystemExit(
            "S3 access_key / secret_key are not set. "
            "Use --access-key / --secret-key or env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
        )

    config.root_prefix = config.root_prefix or args.root_prefix
    if not config:
        raise SystemExit(
            "Root Prefix is not set. Use --root-prefix or env CCLANG_S3_ROOT_PREFIX"
        )
    return config


def _build_file_manager(args: argparse.Namespace) -> tuple[FileManager, Path]:
    """Construct FileManager with local and cloud configuration."""
    data_path = _resolve_data_path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    local_cfg = LocalConfig(
        base_path=data_path,
        save_local=True,
        cache_files=True,
        temp_base=data_path / "temp",
    )

    s3_cfg = _build_s3_config_from_env_and_args(args)
    upload_threads = max(0, args.upload_threads)
    max_pool_connections = args.s3_max_pool_connections
    if max_pool_connections is None:
        max_pool_connections = max(32, upload_threads * 4)

    cloud_cfg = CloudConfig(
        enable=True,
        base_path=Path(args.remote_base),  # prefix inside root_prefix in bucket, e.g. "data"
        s3_config=s3_cfg,
        max_upload_threads=upload_threads,
        max_pool_connections=max_pool_connections,
    )

    fm = FileManager(local_cfg=local_cfg, cloud_cfg=cloud_cfg)
    return fm, data_path


def _iter_files(base: Path, subdirs: Sequence[str]) -> Iterable[Path]:
    """Yield all files under the given subdirectories."""
    for sub in subdirs:
        root = base / sub
        if not root.exists():
            logger.warning("Subdir %s does not exist under %s, skipping", sub, base)
            continue

        for path in root.rglob("*"):
            if path.is_file():
                if path.name == ".DS_Store":
                    continue
                yield path


def _upload_one(
        fm: FileManager,
        data_path: Path,
        local_path: Path,
        dry_run: bool,
        skip_existing: bool,
) -> None:
    """
    Upload a single file to S3, preserving the path relative to data_path.
    """
    relative = local_path.relative_to(data_path)

    if skip_existing and fm.exists_cloud(relative):
        logger.info("Skip existing in S3: %s", relative.as_posix())
        return

    if dry_run:
        logger.info("DRY RUN: would upload %s", relative.as_posix())
        return

    if not fm.cloud:
        raise RuntimeError("Cloud store is not enabled in FileManager")

    logger.info("Uploading %s", relative.as_posix())
    log_cb = LoggingCallBack(relative.as_posix(), logger)
    fm.cloud_upload(local_path, relative, blocking=False, callback=log_cb)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for upload-local-to-s3 CLI."""
    parser = argparse.ArgumentParser(
        description="Upload local data/ subdirectories to S3 (Yandex Object Storage)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Local data root (default: CCLANG_DATA_DIR)",
    )
    parser.add_argument(
        "--remote-base",
        type=str,
        default="data",
        help="Prefix inside S3 (relative to root_prefix), default 'data'",
    )
    parser.add_argument(
        "--upload-threads",
        type=int,
        default=16,
        help="Number of upload threads (0 — synchronous, default 16)",
    )
    parser.add_argument(
        "--s3-max-pool-connections",
        type=int,
        help="Explicit max_pool_connections for boto3; by default 4 per thread, minimum 32",
    )

    parser.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        help=(
            "Which data/ subdirectories to upload. "
            "Can be provided multiple times. Default: raw_pdfs, corpora, artifacts."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not upload anything, only print what would be done",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Existing objects in S3 are skipped by default. "
             "Set this flag to overwrite everything.",
    )

    # S3 / Yandex Object Storage parameters
    parser.add_argument("--bucket", type=str, help="S3 bucket (or env CCLANG_S3_BUCKET)")
    parser.add_argument("--region", type=str, help="S3 region (or env AWS_REGION, default ru-central1)")
    parser.add_argument("--access-key", type=str, help="S3 access key (or env AWS_ACCESS_KEY_ID)")
    parser.add_argument("--secret-key", type=str, help="S3 secret key (or env AWS_SECRET_ACCESS_KEY)")
    parser.add_argument("--root-prefix", type=str, help="Root prefix in bucket (or env CCLANG_S3_ROOT_PREFIX)")
    parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            required=False, help='level of logging')

    args = parser.parse_args(argv)

    logx.setup_logging(service="script:upload_s3", level=args.log_level)

    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        if stop_event.is_set():
            return
        logger.warning("stop requested", extra={"signal": sig})
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass

    dirs = args.dirs if args.dirs else list(DEFAULT_SUBDIRS)

    fm, data_path = _build_file_manager(args)

    logger.info("Data root: %s", data_path)
    logger.info(
        "Uploading subdirs: %s (remote base: %s)",
        ", ".join(dirs),
        fm.cloud_cfg.base_path.as_posix(),
    )

    total = 0
    try:
        for local_path in _iter_files(data_path, dirs):
            if stop_event.is_set():
                logger.info("Stop requested, cancelling remaining uploads")
                break
            _upload_one(
                fm=fm,
                data_path=data_path,
                local_path=local_path,
                dry_run=args.dry_run,
                skip_existing=not args.no_skip_existing,
            )
            total += 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, stopping uploads")
    finally:
        fm.close()
        logger.info("Done. Processed %d local files.", total)


if __name__ == "__main__":
    main()
