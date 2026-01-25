import sys
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable
from itertools import islice

from cclang.common import logx
from cclang.config.s3 import load_s3_config
from cclang.io.pdf_state_store import PdfStateStore
from cclang.io.db import get_conn
from cclang.ingest.fs import FileManager, LocalConfig, CloudConfig, ensure_relative
from cclang.io.schemas import ProcessingStatus

logger = logx.get_logger(__name__)


def get_pdfs_path(pdf_state: PdfStateStore)->Iterable[Path]:
    rows = pdf_state.filter_by_status()
    return map(lambda row: Path(row.pdf_path), filter(lambda row: row.pdf_path is not None, rows))

def get_raw_text_path(pdf_state: PdfStateStore)->Iterable[Path]:
    rows = pdf_state.filter_by_text_status(ProcessingStatus.OK)
    return map(lambda row: Path(row.text_path), filter(lambda row: row.text_path is not None, rows))

def get_tokenize_text_path(pdf_state: PdfStateStore)->Iterable[Path]:
    rows = pdf_state.filter_by_tokenize_status(ProcessingStatus.OK)
    return map(lambda row: Path(row.tokenize_path), filter(lambda row: row.tokenize_path is not None, rows))

parser_targeting = {
    'pdf': get_pdfs_path,
    'raw': get_raw_text_path,
    'tokenize': get_tokenize_text_path,
}

def _resolve_data_path(cli_value: Path | None) -> Path:
    """
    Determine local data root:
    1) CLI-provided path if set;
    2) env CCLANG_DATA_DIR if set;
    3) fallback to ./data
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env_val = os.environ.get("CCLANG_DATA_DIR")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return Path("data").resolve()


def main(argv: Iterable[str] | None = None) -> None:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--database-dsn", "-db", dest="db_dsn", default=None)
    arg_parser.add_argument("--data-path", "-dp", dest="data_path", type=Path, default=None)
    arg_parser.add_argument('-head', dest="head", default=None, type=int)
    arg_parser.add_argument('-stage', dest='stage', choices=['pdf', 'raw', 'tokenize'], required=True)
    arg_parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            required=False, help='level of logging')
    arg_parser.add_argument('--overwrite', action='store_true', help='Redownload even if local file already exists')

    args = arg_parser.parse_args(argv)
    logx.setup_logging(service="script:load_local_from_s3", level=args.log_level)

    db_dsn = args.db_dsn
    data_path = _resolve_data_path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    s3_config = load_s3_config()
    if not s3_config.enable:
        raise RuntimeError("S3 access is disabled. Set CCLANG_S3_ENABLE=true to download files.")

    db_conn = get_conn(db_dsn)
    pdf_state = PdfStateStore(db_conn)
    local_conf = LocalConfig(data_path, True, True, data_path / "temp/load_local_from_s3")
    cloud_conf = CloudConfig(s3_config.enable, base_path=Path("data"), max_upload_threads=2, s3_config=s3_config)
    file_manager = FileManager(local_conf, cloud_conf)

    logger.info("Starting download", extra={"stage": args.stage, "data_path": str(data_path), "overwrite": args.overwrite})

    target_paths = list(islice(parser_targeting[args.stage](pdf_state), args.head))
    logger.info("Total targets resolved", extra={"count": len(target_paths)})

    for target_rel_path in target_paths:
        try:
            normalized_rel = ensure_relative(Path(str(target_rel_path).replace("\\", "/")), data_path, "target_path")
            local_destination = data_path / normalized_rel
            if local_destination.exists() and not args.overwrite:
                logger.info("Skip existing file", extra={"path": str(normalized_rel)})
                continue

            logger.info("Downloading from cloud", extra={"path": str(normalized_rel)})
            downloaded_path = file_manager.fetch_from_cloud(normalized_rel, overwrite=args.overwrite)
            logger.info("Downloaded", extra={"path": str(downloaded_path.relative_to(data_path))})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to download file", exc=exc, extra={"path": str(target_rel_path)})

    file_manager.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
