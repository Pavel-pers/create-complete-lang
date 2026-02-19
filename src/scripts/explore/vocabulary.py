"""CLI to compute vocabulary and lemma statistics from the corpus."""

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
from cclang.corpus.corpus_preprocessing import (
    classify_lemma,
    is_ocr_garbage,
    is_valid_devanagari,
    normalize_devanagari,
)
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.fs import LocalConfig
from cclang.io.doc_state_store import DocStateStore
from cclang.io.schemas import DocLemma

logger = logx.get_logger(__name__)


# ---------------------------------------------------------------------------
# Statistics collection
# ---------------------------------------------------------------------------

def collect_stats(
        data_path: Path,
        db_dsn: str | None,
        lemmatizer_name: str = None,
        data_preparation: str = None,
        top_n: int = 5,
        min_df: int | float = 2,
        max_df: float = 0.90,
        head: int | None = None,
        stop_event: threading.Event | None = None,
        ignore_oov: bool = False,
        enable_normalization: bool = False,
) -> dict[str, Any]:
    """
    Collect vocabulary statistics from all lemmatized documents.

    Args:
        data_path: Root data directory containing corpora
        db_dsn: PostgreSQL DSN for querying document state
        lemmatizer_name: Name of lemmatizer method (default apertium)
        data_preparation: Preparation method (default none)
        top_n: Number of top lemmas to include
        min_df: Minimum document frequency. If int, absolute count.
                If float in [0.0, 1.0], proportion of documents.
        max_df: Maximum document frequency as proportion [0.0, 1.0].
                Lemmas appearing in more documents are filtered out.
        head: Limit number of documents to process (for testing)
        stop_event: Optional event to signal early termination
        ignore_oov: If True, ignore OOV tokens
        enable_normalization: If True, apply Unicode NFC normalization and
                              Devanagari validity / OCR-garbage filtering.
    Returns:
        Dictionary with vocabulary statistics
    """
    if lemmatizer_name is None:
        lemmatizer_name = "apertium-mar-morph"

    if data_preparation is None:
        data_preparation = 'none'

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
    lemma_tf: Counter[str] = Counter()  # Term frequency (total occurrences)
    lemma_df: Counter[str] = Counter()  # Document frequency (in how many docs)
    ner_counter: Counter[str] = Counter()  # Ner tags counter
    unique_lemmas_per_doc: list[int] = []

    # ---- normalization statistics ----
    norm_stats: dict[str, int] = {
        'tokens_before_norm': 0,          # total tokens seen
        'tokens_normalized_changed': 0,   # NFC changed the surface form
        'unique_collapsed_by_nfc': 0,     # filled after full pass
        'rejected_non_devanagari': 0,
        'rejected_invalid_deva': 0,
        'rejected_ocr_garbage': 0,
        'tokens_accepted': 0,
    }

    # When normalization is on we also keep raw→normalized map to measure
    # how many unique lemmas NFC collapses.
    raw_lemma_set: set[str] = set()  # unique raw lemmas (before NFC)

    try:
        # Query all documents with completed lemma results
        docs = doc_store.get_completed_lemma_results(method=lemmatizer_name)

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
            doc_lemmas: set[str] = set()
            for sentence in doc_lemma.sentences:
                for token in sentence:
                    if not ignore_oov or not token.is_oov:
                        lemma = token.lemma

                        if token.is_ne:
                            ner_counter[str(token.ner_result)] += 1
                            if data_preparation == 'replace_ner':
                                lemma = '<' + str(token.ner_result).upper() + '>'

                        # ---------- normalization pipeline ----------
                        if enable_normalization:
                            norm_stats['tokens_before_norm'] += 1
                            raw_lemma_set.add(lemma)

                            # Skip NER-replaced tokens from filtering
                            is_special = lemma.startswith('<') and lemma.endswith('>')
                            if not is_special:
                                # 1. NFC normalization
                                normalized = normalize_devanagari(lemma)
                                if normalized != lemma:
                                    norm_stats['tokens_normalized_changed'] += 1
                                lemma = normalized

                                # 2. Validity & OCR checks
                                classification = classify_lemma(lemma)
                                if classification == 'non_devanagari':
                                    norm_stats['rejected_non_devanagari'] += 1
                                    continue
                                elif classification == 'invalid_deva':
                                    norm_stats['rejected_invalid_deva'] += 1
                                    continue
                                elif classification == 'ocr_garbage':
                                    norm_stats['rejected_ocr_garbage'] += 1
                                    continue

                            norm_stats['tokens_accepted'] += 1
                        # ---------- end normalization pipeline ------

                        total_tokens += 1
                        lemma_tf[lemma] += 1
                        doc_lemmas.add(lemma)

            # Update document frequency for each unique lemma in this doc
            for lemma in sorted(doc_lemmas):
                lemma_df[lemma] += 1

            unique_lemmas_per_doc.append(len(doc_lemmas))
            total_documents += 1

            if total_documents % 10 == 0:
                logger.info(
                    "Processing documents",
                    extra={
                        "processed": total_documents,
                        "total_tokens": total_tokens,
                        "unique_lemmas": len(lemma_tf),
                    },
                )

    finally:
        storage.close()
        doc_store.close()

    # Measure NFC collapse (unique raw forms → unique normalized forms)
    if enable_normalization and raw_lemma_set:
        normalized_set = {normalize_devanagari(l) for l in raw_lemma_set}
        norm_stats['unique_collapsed_by_nfc'] = len(raw_lemma_set) - len(normalized_set)

    # Convert min_df to absolute count if it's a proportion
    if isinstance(min_df, float) and 0.0 <= min_df <= 1.0:
        min_df_abs = int(min_df * total_documents)
    else:
        min_df_abs = int(min_df)

    # Convert max_df to absolute count
    max_df_abs = int(max_df * total_documents)

    # Ensure reasonable bounds
    min_df_abs = max(1, min_df_abs)
    max_df_abs = max(min_df_abs, max_df_abs)

    logger.info(
        "Applying DF filters",
        extra={
            "min_df": min_df_abs,
            "max_df": max_df_abs,
            "total_documents": total_documents,
        },
    )

    # Filter lemmas by document frequency
    total_lemmas_before = len(lemma_tf)
    filtered_by_min_df = 0
    filtered_by_max_df = 0
    filtered_lemma_tf: Counter[str] = Counter()
    filtered_lemma_df: Counter[str] = Counter()

    for lemma, df in lemma_df.items():
        if df < min_df_abs:
            filtered_by_min_df += 1
        elif df > max_df_abs:
            filtered_by_max_df += 1
        else:
            filtered_lemma_tf[lemma] = lemma_tf[lemma]
            filtered_lemma_df[lemma] = df

    total_lemmas_after = len(filtered_lemma_tf)

    # Compute tokens after filtering
    tokens_after_filter = sum(filtered_lemma_tf.values())

    # Compute final statistics
    type_token_ratio_raw = len(lemma_tf) / total_tokens if total_tokens > 0 else 0.0
    type_token_ratio_filtered = (
        total_lemmas_after / tokens_after_filter if tokens_after_filter > 0 else 0.0
    )

    avg_unique_per_doc = (
        sum(unique_lemmas_per_doc) / len(unique_lemmas_per_doc)
        if unique_lemmas_per_doc
        else 0.0
    )
    min_unique_per_doc = min(unique_lemmas_per_doc) if unique_lemmas_per_doc else 0
    max_unique_per_doc = max(unique_lemmas_per_doc) if unique_lemmas_per_doc else 0

    # Get top N lemmas (from filtered set)
    top_lemmas = [
        {"lemma": lemma, "tf": tf, "df": filtered_lemma_df[lemma]}
        for lemma, tf in filtered_lemma_tf.most_common(top_n)
    ]

    # Get examples of filtered lemmas
    rare_lemmas_examples = [
        {"lemma": lemma, "tf": lemma_tf[lemma], "df": df}
        for lemma, df in lemma_df.most_common()[:-31:-1]  # 10 rarest
        if df < min_df_abs
    ][:30]

    frequent_lemmas_examples = [
        {"lemma": lemma, "tf": lemma_tf[lemma], "df": df}
        for lemma, df in lemma_df.most_common(20)
        if df > max_df_abs
    ][:5]

    result = {
        "total_documents": total_documents,
        "total_tokens": total_tokens,
        # Raw vocabulary stats (before filtering)
        "vocabulary_raw": {
            "total_unique_lemmas": total_lemmas_before,
            "type_token_ratio": round(type_token_ratio_raw, 3),
        },
        # Document frequency filter settings
        "ner_tags_info": {
                             "total_ner_count": ner_counter.total(),
                             "ner_ratio": round(ner_counter.total() / total_tokens, 3)
                             if total_tokens > 0 else 0,
                         } | {key + '_total_count': count for key, count in ner_counter.items()},
        "df_filter": {
            "min_df": min_df_abs,
            "min_df_ratio": round(min_df_abs / total_documents, 4) if total_documents > 0 else 0,
            "max_df": max_df_abs,
            "max_df_ratio": round(max_df_abs / total_documents, 4) if total_documents > 0 else 0,
            "filtered_by_min_df": filtered_by_min_df,
            "filtered_by_max_df": filtered_by_max_df,
            "total_filtered": filtered_by_min_df + filtered_by_max_df,
            "rare_lemmas_examples": rare_lemmas_examples,
            "frequent_lemmas_examples": frequent_lemmas_examples,
        },
        # Filtered vocabulary stats
        "vocabulary_filtered": {
            "total_unique_lemmas": total_lemmas_after,
            "tokens_covered": tokens_after_filter,
            "tokens_coverage_ratio": round(tokens_after_filter / total_tokens, 4)
            if total_tokens > 0
            else 0,
            "type_token_ratio": round(type_token_ratio_filtered, 6),
        },
        # Per-document stats
        "per_document": {
            "avg_unique_lemmas": round(avg_unique_per_doc, 2),
            "min_unique_lemmas": min_unique_per_doc,
            "max_unique_lemmas": max_unique_per_doc,
        },
        # Top lemmas from filtered vocabulary
        "top_lemmas": top_lemmas,
    }

    # ---- append normalization report if enabled ----
    if enable_normalization:
        total_rejected = (
            norm_stats['rejected_non_devanagari']
            + norm_stats['rejected_invalid_deva']
            + norm_stats['rejected_ocr_garbage']
        )
        result["normalization"] = {
            "enabled": True,
            "tokens_before_norm": norm_stats['tokens_before_norm'],
            "tokens_accepted": norm_stats['tokens_accepted'],
            "tokens_rejected_total": total_rejected,
            "tokens_rejected_ratio": (
                round(total_rejected / norm_stats['tokens_before_norm'], 4)
                if norm_stats['tokens_before_norm'] > 0 else 0
            ),
            "breakdown": {
                "nfc_changed_tokens": norm_stats['tokens_normalized_changed'],
                "unique_lemmas_collapsed_by_nfc": norm_stats['unique_collapsed_by_nfc'],
                "rejected_non_devanagari": norm_stats['rejected_non_devanagari'],
                "rejected_invalid_devanagari": norm_stats['rejected_invalid_deva'],
                "rejected_ocr_garbage": norm_stats['rejected_ocr_garbage'],
            },
        }
    else:
        result["normalization"] = {"enabled": False}

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for vocabulary statistics."""
    parser = argparse.ArgumentParser(
        description="Compute vocabulary and lemma statistics from the corpus."
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
    parser.add_argument('--method', choices=['apertium', 'stanza'], default='apertium',
                        help="Method of lemmatization which to consider (default: apertium)")
    parser.add_argument(
        '--data-preparation',
        choices=['none', 'replace_ner'],
        default='none'
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top lemmas to include (default: 5)",
    )
    parser.add_argument(
        "--min-df",
        type=float,
        default=2,
        help="Min document frequency. Int for absolute count, float [0-1] for proportion. "
             "Lemmas in fewer docs are filtered out. (default: 2)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.90,
        help="Max document frequency as proportion [0-1]. "
             "Lemmas in more docs are filtered out as stopwords. (default: 0.90)",
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
    parser.add_argument(
        "--ignore-oov",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Enable Unicode NFC normalization and Devanagari OCR filtering. "
             "Rejected tokens are excluded from vocabulary; statistics are "
             "reported in the 'normalization' section of the output.",
    )

    lemmatizer_name_map = {
        "apertium": "apertium-mar-morph",
        "stanza": "stanza-marathi",
    }

    args = parser.parse_args(argv)

    lemmatizer_name = lemmatizer_name_map[args.method]
    logx.setup_logging(service="script:vocabulary", level=args.log_level)

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

    # Parse min_df: if it looks like proportion (has decimal), keep as float
    min_df: int | float
    if args.min_df < 1.0:
        min_df = args.min_df  # proportion
    else:
        min_df = int(args.min_df)  # absolute count

    try:
        stats = collect_stats(
            data_path=args.data_path,
            db_dsn=db_dsn,
            lemmatizer_name=lemmatizer_name,
            data_preparation=args.data_preparation,
            top_n=args.top_n,
            min_df=min_df,
            max_df=args.max_df,
            head=args.head,
            stop_event=stop_event,
            ignore_oov=args.ignore_oov,
            enable_normalization=args.normalize,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as err:
        logger.exception("Unexpected error", extra={"exc": str(err)})
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
    