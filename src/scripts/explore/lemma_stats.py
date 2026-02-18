"""CLI to compute lemmatization quality statistics from the corpus."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from cclang.common import logx
from cclang.config.s3 import load_s3_config
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.fs import LocalConfig
from cclang.io.doc_state_store import DocStateStore
from cclang.io.schemas import DocLemma

logger = logx.get_logger(__name__)


def collect_stats(
    data_path: Path,
    db_dsn: str | None,
    head: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """
    Collect lemmatization quality statistics from all lemmatized documents.

    Args:
        data_path: Root data directory containing corpora
        db_dsn: PostgreSQL DSN for querying document state
        head: Limit number of documents to process (for testing)
        stop_event: Optional event to signal early termination

    Returns:
        Dictionary with lemmatization quality statistics
    """
    if stop_event is None:
        stop_event = threading.Event()

    # Setup storage manager (local only, no cloud uploads needed)
    local_cfg = LocalConfig(
        base_path=data_path,
        save_local=True,
        cache_files=True,
        temp_base=data_path / "temp",
    )
    s3_cfg = load_s3_config()
    cloud_cfg = CloudConfig(
        enable=s3_cfg.enable,
        base_path=Path("data"),
        s3_config=s3_cfg,
        max_upload_threads=0,
    )
    storage = StorageManager(local_cfg=local_cfg, cloud_cfg=cloud_cfg)

    # Connect to database
    db_conn = get_conn(db_dsn)
    doc_store = DocStateStore(db_conn)

    # Statistics accumulators
    total_tokens = 0
    total_documents = 0
    oov_count = 0
    ambiguous_count = 0
    total_analyses = 0
    pos_counter: Counter[str] = Counter()

    try:
        # Query all documents with completed lemma results
        docs = doc_store.get_completed_lemma_results()

        for result in docs:
            if stop_event.is_set():
                logger.warning("Stop requested, returning partial results")
                break

            if head is not None and total_documents >= head:
                logger.info("Reached document limit", extra={"head": head})
                break

            lemma_path = result.path
            if not lemma_path:
                logger.warning(
                    "Document has no lemma_path",
                    extra={"doc_id": result.doc_id},
                )
                continue

            try:
                with storage.open(lemma_path, mode="r", encoding="utf-8") as f:
                    doc_lemma = DocLemma.model_validate_json(f.read())
            except FileNotFoundError:
                logger.warning(
                    "Lemma file not found",
                    extra={"lemma_path": lemma_path, "doc_id": result.doc_id},
                )
                continue
            except Exception as e:
                logger.error(
                    "Failed to load lemma file",
                    extra={"lemma_path": lemma_path, "error": str(e)},
                )
                continue

            # Collect statistics from this document
            for sentence in doc_lemma.sentences:
                for token in sentence:
                    total_tokens += 1

                    if token.is_oov:
                        oov_count += 1

                    if token.is_ambiguous:
                        ambiguous_count += 1

                    # POS distribution - use "null" for missing POS
                    pos_tag = token.pos if token.pos else "null"
                    pos_counter[pos_tag] += 1

                    # Count analyses for morphological richness
                    if token.analyses:
                        total_analyses += len(token.analyses)
                    else:
                        # Token with no analyses counts as 1 (the lemma itself)
                        total_analyses += 1

            total_documents += 1

            if total_documents % 10 == 0:
                logger.info(
                    "Processing documents",
                    extra={
                        "processed": total_documents,
                        "total_tokens": total_tokens,
                        "oov_count": oov_count,
                    },
                )

    finally:
        storage.close()
        doc_store.close()

    # Compute final statistics
    oov_rate = oov_count / total_tokens if total_tokens > 0 else 0.0
    ambiguity_rate = ambiguous_count / total_tokens if total_tokens > 0 else 0.0
    avg_analyses_per_token = total_analyses / total_tokens if total_tokens > 0 else 0.0

    # Sort POS distribution by count (descending)
    pos_distribution = dict(pos_counter.most_common())

    return {
        "total_documents": total_documents,
        "total_tokens": total_tokens,
        "oov_count": oov_count,
        "oov_rate": round(oov_rate, 4),
        "ambiguous_count": ambiguous_count,
        "ambiguity_rate": round(ambiguity_rate, 4),
        "pos_distribution": pos_distribution,
        "avg_analyses_per_token": round(avg_analyses_per_token, 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for lemma quality statistics."""
    parser = argparse.ArgumentParser(
        description="Compute lemmatization quality statistics from the corpus."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--database-dsn",
        "-db",
        type=str,
        default=None,
        help="PostgreSQL DSN (or env CCLANG_DB_DSN)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)

    logx.setup_logging(service="script:lemma_stats", level=args.log_level)

    # Handle graceful shutdown
    stop_event = threading.Event()

    def _handle_signal(sig: int, _frame: Any) -> None:
        if not stop_event.is_set():
            logger.warning("Stop requested", extra={"signal": sig})
            stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except (OSError, ValueError):
                pass

    db_dsn = args.database_dsn or os.environ.get("CCLANG_DB_DSN")
    if not db_dsn:
        logger.error("Database DSN not provided. Use --database-dsn or set CCLANG_DB_DSN")
        return 1

    try:
        stats = collect_stats(
            data_path=args.data_path,
            db_dsn=db_dsn,
            head=args.head,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Unexpected error")
        return 1

    # Output results
    output_json = json.dumps(stats, indent=2, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        logger.info("Statistics written", extra={"output": str(args.output)})
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
