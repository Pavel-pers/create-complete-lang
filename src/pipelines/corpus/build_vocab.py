"""Pipeline: build vocabulary (.tsv) from lemmatized documents."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.corpus.corpus_preprocessing import (
    is_oov,
    normalize_lemma,
    replace_ner,
)
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.fs import LocalConfig
from cclang.io.schemas import DocLemma, ProcessingStatus, VocabParams, VocabStats

LEMMATIZER_NAME_MAP = {
    "apertium": "apertium-mar-morph",
    "stanza": "stanza-marathi",
}


def run_pipeline(
    data_path: Path,
    database_dsn: str | None,
    method: str,
    output_path: str,
    min_df: int | float,
    max_df: float,
    min_tf: int,
    do_replace_ner: bool,
    do_normalize: bool,
    do_ignore_oov: bool,
    head: int | None,
    log: BoundLogger,
    stop_event: threading.Event,
) -> int:
    """Build a vocabulary TSV from completed lemmatizations."""
    lemmatizer_name = LEMMATIZER_NAME_MAP[method]

    # Storage manager (local only, no upload threads needed)
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

    db_conn = get_conn(database_dsn)
    doc_store = DocStateStore(db_conn)

    # Accumulators
    lemma_tf: Counter[str] = Counter()
    lemma_df: Counter[str] = Counter()
    total_documents = 0

    try:
        docs = doc_store.get_completed_lemma_results(method=lemmatizer_name)
        log.info("documents found", extra={"count": len(docs), "method": lemmatizer_name})

        for result in docs:
            if stop_event.is_set():
                log.warning("stop requested, aborting with partial data")
                break

            if head is not None and total_documents >= head:
                log.info("reached document limit", extra={"head": head})
                break

            lemma_path = result.path
            if not lemma_path:
                log.warning("document has no lemma_path", extra={"doc_id": result.doc_id})
                continue

            try:
                with storage.open(Path(lemma_path), mode="r", encoding="utf-8") as f:
                    doc_lemma = DocLemma.model_validate_json(f.read())
            except FileNotFoundError:
                log.warning("lemma file not found", extra={"path": lemma_path})
                continue
            except Exception as e:
                log.error("failed to load lemma file", extra={"path": lemma_path, "error": str(e)})
                continue

            doc_lemmas: set[str] = set()
            for sentence in doc_lemma.sentences:
                for token in sentence:
                    if do_ignore_oov and is_oov(token):
                        continue

                    lemma = replace_ner(token) if do_replace_ner else token.lemma

                    if do_normalize:
                        normalized = normalize_lemma(lemma)
                        if normalized is None:
                            continue
                        lemma = normalized

                    lemma_tf[lemma] += 1
                    doc_lemmas.add(lemma)

            for lemma in doc_lemmas:
                lemma_df[lemma] += 1

            total_documents += 1

            if total_documents % 10 == 0:
                log.info(
                    "progress",
                    extra={
                        "processed": total_documents,
                        "unique_lemmas": len(lemma_tf),
                    },
                )

        if total_documents == 0:
            log.error("no documents processed, cannot build vocabulary")
            return 1

        # --- Apply frequency filters ---

        # Convert min_df
        if isinstance(min_df, float) and 0.0 <= min_df < 1.0:
            min_df_abs = max(1, int(min_df * total_documents))
        else:
            min_df_abs = int(min_df)

        # Convert max_df
        if isinstance(max_df, float) and 0.0 < max_df <= 1.0:
            max_df_abs = int(max_df * total_documents)
        else:
            max_df_abs = int(max_df)

        max_df_abs = max(min_df_abs, max_df_abs)

        log.info(
            "applying filters",
            extra={
                "min_df_abs": min_df_abs,
                "max_df_abs": max_df_abs,
                "min_tf": min_tf,
                "total_documents": total_documents,
                "unique_lemmas_before": len(lemma_tf),
            },
        )

        filtered: dict[str, tuple[int, int]] = {}
        for lemma, df in lemma_df.items():
            tf = lemma_tf[lemma]
            if df < min_df_abs:
                continue
            if df > max_df_abs:
                continue
            if tf < min_tf:
                continue
            filtered[lemma] = (tf, df)

        log.info("after filtering", extra={"unique_lemmas": len(filtered)})

        # Sort by tf descending, assign idx
        sorted_vocab = sorted(filtered.items(), key=lambda x: x[1][0], reverse=True)

        # --- Write TSV artifact ---
        temp_file = storage.create_temp_file(".vocab.tsv")
        try:
            with open(temp_file, mode="w", encoding="utf-8") as fout:
                fout.write("idx\tlemma\ttf\tdf\n")
                for idx, (lemma, (tf, df)) in enumerate(sorted_vocab):
                    fout.write(f"{idx}\t{lemma}\t{tf}\t{df}\n")

            # Compute sha256
            sha = hashlib.sha256(temp_file.read_bytes()).hexdigest()

            # Finalize artifact
            dest = Path(output_path)
            storage.finalize_artifact(temp_file, dest, blocking=True)
            log.info(
                "vocabulary written",
                extra={"path": str(dest), "sha256": sha, "vocab_size": len(sorted_vocab)},
            )
        except Exception:
            # Clean up temp file on error
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        # --- Record in DB ---
        vocab_params = VocabParams(
            min_df=min_df_abs,
            max_df=max_df_abs,
            min_tf=min_tf,
            replace_ner=do_replace_ner,
            normalize=do_normalize,
            ignore_oov=do_ignore_oov,
        )
        vocab_stats = VocabStats(
            total_lemmas=len(sorted_vocab),
            total_documents=total_documents,
        )
        run_id = doc_store.insert_vocab_build(
            lemma_method=lemmatizer_name,
            artefact_id=sha,
            status=ProcessingStatus.OK,
            path=str(dest),
            params=vocab_params,
            stats=vocab_stats,
        )
        log.info("vocab build recorded", extra={"run_id": run_id})

    finally:
        storage.close()
        doc_store.close()
        log.info("resources closed")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for build_vocab pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build vocabulary TSV from lemmatized documents."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Root data directory (env: CCLANG_DATA_DIR, default: data)",
    )
    parser.add_argument(
        "--database-dsn",
        "-db",
        type=str,
        default=None,
        help="PostgreSQL DSN (or env CCLANG_DB_DSN)",
    )
    parser.add_argument(
        "--method",
        choices=["apertium", "stanza"],
        default="apertium",
        help="Lemmatization method to consider (default: apertium)",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=True,
        help="Output .tsv file path (relative to data-path)",
    )
    parser.add_argument(
        "--min-df",
        type=float,
        default=2,
        help="Min document frequency. Int = absolute, float [0,1) = proportion. (default: 2)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.90,
        help="Max document frequency as proportion [0,1]. (default: 0.90)",
    )
    parser.add_argument(
        "--min-tf",
        type=int,
        default=1,
        help="Min term frequency, absolute. (default: 1)",
    )
    parser.add_argument(
        "--replace-ner",
        action="store_true",
        default=False,
        help="Replace NE tokens with <TAG>",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply NFC normalization + Devanagari/OCR filtering",
    )
    parser.add_argument(
        "--ignore-oov",
        action="store_true",
        default=False,
        help="Skip OOV tokens",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Limit number of documents (for testing)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--log-format",
        choices=["console", "json"],
        default="console",
    )
    parser.add_argument("--log-file", default=None, help="File for logs (JSONL)")

    args = parser.parse_args(argv)

    data_path = args.data_path or Path(os.environ.get("CCLANG_DATA_DIR", "data"))
    database_dsn = args.database_dsn or os.environ.get("CCLANG_DB_DSN")

    log_file = Path(args.log_file or data_path / "logs/corpus/build_vocab.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(fmt=args.log_format, level=args.log_level, file=str(log_file))
    log = get_logger("root").bind(pipeline="build_vocab")

    if not database_dsn:
        log.error("database DSN not provided. Use --database-dsn or set CCLANG_DB_DSN")
        return 1

    # Parse min_df: float < 1.0 → proportion, else absolute
    min_df: int | float
    if args.min_df < 1.0:
        min_df = args.min_df
    else:
        min_df = int(args.min_df)

    # Graceful shutdown
    stop_event = threading.Event()

    def _handle_signal(sig: int, _frame: Any) -> None:
        if not stop_event.is_set():
            log.warning("received stop signal", extra={"signal": sig})
            stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except (OSError, ValueError):
                pass

    try:
        return run_pipeline(
            data_path=data_path,
            database_dsn=database_dsn,
            method=args.method,
            output_path=args.output_path,
            min_df=min_df,
            max_df=args.max_df,
            min_tf=args.min_tf,
            do_replace_ner=args.replace_ner,
            do_normalize=args.normalize,
            do_ignore_oov=args.ignore_oov,
            head=args.head,
            log=log,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        return 130
    except Exception:
        log.exception("unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
