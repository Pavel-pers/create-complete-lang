"""Pipeline: index corpus – split lemmatized docs into fragments and map to vocab indices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
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
from cclang.io.fs import LocalConfig, get_shard_relative
from cclang.io.schemas import DocLemma, FragmentProcessed, LemmaToken, ProcessingStatus


def load_vocab_tsv(storage: StorageManager, tsv_path: Path) -> dict[str, int]:
    """Read a vocab .tsv (idx, lemma, tf, df) and return {lemma: idx}."""
    lemma_to_idx: dict[str, int] = {}
    with storage.open(tsv_path, mode="r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue  # skip header
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            lemma = parts[1]
            lemma_to_idx[lemma] = idx
    return lemma_to_idx


def process_token(
    token: LemmaToken,
    do_replace_ner: bool,
    do_normalize: bool,
    do_ignore_oov: bool,
) -> str | None:
    """Apply preprocessing to a single token, return lemma or None (skip)."""
    if do_ignore_oov and is_oov(token):
        return None

    lemma = replace_ner(token) if do_replace_ner else token.lemma

    if do_normalize:
        normalized = normalize_lemma(lemma)
        if normalized is None:
            return None
        lemma = normalized

    return lemma


def run_pipeline(
    data_path: Path,
    database_dsn: str | None,
    vocab_id: int,
    fragment_size: int,
    output_base_path: str,
    head: int | None,
    log: BoundLogger,
    stop_event: threading.Event,
) -> int:
    """Index corpus: split lemmatized docs into fragments mapped to vocab indices."""
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

    build_id: int | None = None

    try:
        # 1. Load vocab build info
        vocab_info = doc_store.get_vocab_build(vocab_id)
        log.info(
            "vocab build loaded",
            extra={"vocab_id": vocab_id, "lemma_method": vocab_info.lemma_method},
        )

        # 2. Load vocab .tsv
        if not vocab_info.path:
            log.error("vocab build has no path", extra={"vocab_id": vocab_id})
            return 1
        lemma_to_idx = load_vocab_tsv(storage, Path(vocab_info.path))
        log.info("vocabulary loaded", extra={"vocab_size": len(lemma_to_idx)})

        # 3. Extract preprocessing flags from vocab params
        vp = vocab_info.vocab_params
        do_replace_ner = bool(vp and vp.replace_ner)
        do_normalize = bool(vp and vp.normalize)
        do_ignore_oov = bool(vp and vp.ignore_oov)

        # 4. Get completed lemma documents
        lemma_method = vocab_info.lemma_method
        docs = doc_store.get_completed_lemma_results(method=lemma_method)
        log.info("lemma documents found", extra={"count": len(docs), "method": lemma_method})

        # 5. Reserve build_id
        build_id = doc_store.insert_index_build(
            vocab_id=vocab_id,
            fragment_size=fragment_size,
            status=ProcessingStatus.RUNNING,
        )
        log.info("index build created", extra={"build_id": build_id})

        # Stats accumulators
        total_documents = 0
        total_fragments = 0
        total_tokens = 0
        skipped_tail_fragments = 0
        empty_fragments_skipped = 0
        fragment_rows: list[tuple[int, str, str | None, str, str | None]] = []

        output_base_rel = Path(output_base_path)

        # 6. Process each document
        for result in docs:
            if stop_event.is_set():
                log.warning("stop requested, aborting")
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
                log.error(
                    "failed to load lemma file", extra={"path": lemma_path, "error": str(e)}
                )
                continue

            doc_id = result.doc_id

            # 6b. Process sentences → list[list[int]]
            processed_sentences: list[list[int]] = []
            for sentence in doc_lemma.sentences:
                token_ids: list[int] = []
                for token in sentence:
                    lemma = process_token(token, do_replace_ner, do_normalize, do_ignore_oov)
                    if lemma is None:
                        continue
                    idx = lemma_to_idx.get(lemma)
                    if idx is None:
                        continue
                    token_ids.append(idx)
                processed_sentences.append(token_ids)

            # 6d. Chunk into fragments of fragment_size sentences
            n_sentences = len(processed_sentences)
            chunks: list[list[list[int]]] = [
                processed_sentences[i : i + fragment_size]
                for i in range(0, n_sentences, fragment_size)
            ]

            # Drop last chunk if too small
            if len(chunks) > 1 and len(chunks[-1]) < fragment_size // 2:
                skipped_tail_fragments += 1
                chunks = chunks[:-1]

            # 6e. Write each fragment
            position = 0
            for chunk in chunks:
                flat_ids: list[int] = [tid for sent in chunk for tid in sent]
                if not flat_ids:
                    empty_fragments_skipped += 1
                    continue

                fragment_id = hashlib.sha256(
                    f"{build_id}:{doc_id}:{position}:{flat_ids}".encode()
                ).hexdigest()

                fragment = FragmentProcessed(
                    id=fragment_id,
                    source_doc_id=doc_id,
                    position=position,
                    token_ids=flat_ids,
                    vocab_id=vocab_id,
                )

                # Write to sharded path
                shard_rel = output_base_rel / get_shard_relative(fragment_id, ".fragment.json")
                temp_file = storage.create_temp_file(".fragment.json")
                try:
                    with open(temp_file, mode="w", encoding="utf-8") as fout:
                        fout.write(fragment.model_dump_json())
                    storage.finalize_artifact(temp_file, shard_rel, blocking=True)
                except Exception:
                    try:
                        temp_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise

                fragment_rows.append(
                    (build_id, doc_id, fragment_id, "ok", str(shard_rel))
                )
                total_fragments += 1
                total_tokens += len(flat_ids)
                position += 1

            total_documents += 1
            if total_documents % 10 == 0:
                log.info(
                    "progress",
                    extra={"processed": total_documents, "fragments": total_fragments},
                )

        # 7. Stats
        stats = {
            "total_documents": total_documents,
            "total_fragments": total_fragments,
            "total_tokens": total_tokens,
            "skipped_tail_fragments": skipped_tail_fragments,
            "empty_fragments_skipped": empty_fragments_skipped,
        }

        # 8. Batch insert fragment rows
        if fragment_rows:
            doc_store.batch_insert_corpus_fragments(fragment_rows)
            log.info("corpus fragments inserted", extra={"count": len(fragment_rows)})

        # 9. Update build status
        doc_store.update_index_build(build_id, status=ProcessingStatus.OK, stats=stats)
        log.info("index build completed", extra={"build_id": build_id, **stats})

    except Exception:
        if build_id is not None:
            try:
                doc_store.update_index_build(build_id, status=ProcessingStatus.ERROR)
            except Exception:
                pass
        raise
    finally:
        storage.close()
        doc_store.close()
        log.info("resources closed")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for index_corpus pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Index corpus: split lemmatized docs into fragments mapped to vocab indices."
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
        "--vocab-id",
        type=int,
        required=True,
        help="run_id from vocab_builds",
    )
    parser.add_argument(
        "--fragment-size",
        type=int,
        required=True,
        help="Number of sentences per fragment",
    )
    parser.add_argument(
        "--output-base-path",
        "-output",
        type=str,
        required=True,
        help="Base directory for fragment output (relative to data-path)",
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

    log_file = Path(args.log_file or data_path / "logs/corpus/index_corpus.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(fmt=args.log_format, level=args.log_level, file=str(log_file))
    log = get_logger("root").bind(pipeline="index_corpus")

    if not database_dsn:
        log.error("database DSN not provided. Use --database-dsn or set CCLANG_DB_DSN")
        return 1

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
            vocab_id=args.vocab_id,
            fragment_size=args.fragment_size,
            output_base_path=args.output_base_path,
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
