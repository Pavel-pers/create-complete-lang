"""Pipeline that scrapes epustakalay book links and writes them as SourcePDF records."""
import argparse
from typing import Iterable

from requests import RequestException
from requests.exceptions import HTTPError, Timeout, InvalidURL

from cclang.parser.epustakalay.parse_book_list import parse_book_list
from cclang.common.logx import setup_logging, get_logger, use_context, BoundLogger
from cclang.io.schemas import SourcePDF, HttpUrl
from cclang.io.corpora import write_docs
import sys

def run_pipeline(file_path: str, limit: int | None, log: BoundLogger) -> int:
    """Scrape epustakalay links and write them as SourcePDF records to file_path."""
    log.info("pipeline started")
    try:
        book_list = parse_book_list(page_from=1, page_to=89, limit=limit)
    except (HTTPError, Timeout, InvalidURL) as err:
        log.error("network-level error", error=str(err))
        return 2
    except RequestException as err:
        log.error("unexcepted request error", error=str(err))
        return 2

    write_docs(file_path, map(lambda link: SourcePDF(url = HttpUrl(link), lang="mr", source="epustakalay"), book_list))
    log.info("pipeline finished")
    return 0

def main(argv: Iterable[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(
        description="Discover book links of current website"
    )

    arg_parser.add_argument("-path", required=True, help="Path to output directory", type=str)
    arg_parser.add_argument("-limit", default=None, type=int)
    arg_parser.add_argument("--log-format", choices=["console", "json"], default="console")
    arg_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    arg_parser.add_argument("--log-file", default=None, help="Log file path (JSONL)")
    args = arg_parser.parse_args(argv)

    setup_logging(
        service="link-service",
        fmt=args.log_format,
        level=args.log_level,
        file=args.log_file,
    )
    log = get_logger("discover").bind(pipeline="discover_links")

    try:
        return run_pipeline(args.path, args.limit, log)
    except Exception:
        log.exception("unexcepted exception")
        raise

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
