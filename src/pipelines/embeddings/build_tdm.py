"""Pipeline: build Term-Document Matrix from corpus fragments."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import numpy as np
from dotenv import load_dotenv
from scipy import sparse

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.fs import LocalConfig
from cclang.io.schemas import FragmentProcessed, ProcessingStatus


def iter_fragments(
    storage: StorageManager,
    paths: Sequence[str | Path],
    stop_event: threading.Event,
    log: BoundLogger,
) -> Iterator[FragmentProcessed]:
    for i, path in enumerate(paths):
        if stop_event.is_set():
            log.warning("stop requested during fragment iteration")
            return

        path = Path(path)
        with storage.open(path, mode="r") as f:
            json_data = json.load(f)
            fragment = FragmentProcessed.model_validate(json_data)
            yield fragment

        if (i + 1) % 1000 == 0:
            log.info("fragments loaded", extra={"count": i + 1, "total": len(paths)})


def log_entropy_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Log-entropy weighting: local=log(1+tf), global=1-entropy."""
    local = tf_matrix.copy()
    local.data = np.log1p(local.data)

    n_docs = tf_matrix.shape[0]
    tf_sum = np.array(tf_matrix.sum(axis=0)).flatten()

    global_weights = np.ones(tf_matrix.shape[1])
    for token_col_i in range(tf_matrix.shape[1]):
        col = np.array(tf_matrix.getcol(token_col_i).todense()).flatten()
        nnz = col[col > 0]
        if len(nnz) > 0:
            token_p = nnz / tf_sum[token_col_i]
            entropy = -np.sum(token_p * np.log(token_p)) / np.log(n_docs)
            global_weights[token_col_i] = 1.0 - entropy

    return cast(sparse.csr_matrix, local.multiply(global_weights))


def tfidf_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Standard TF-IDF: log(1 + tf) * log(N / df)."""
    n_docs = tf_matrix.shape[0]

    # Local: log(1 + tf)
    local = tf_matrix.copy()
    local.data = np.log1p(local.data)

    # Global: log(N / df)
    df = np.array((tf_matrix > 0).sum(axis=0)).flatten()
    idf = np.log(n_docs / df)

    return cast(sparse.csr_matrix, local.multiply(idf))


WEIGHTING_METHODS_MAPPING = {
    "tf-idf": tfidf_weighting,
    "log-entropy": log_entropy_weighting,
}


def run_pipeline(
    data_path: Path,
    database_dsn: str | None,
    corpus_build_id: int,
    output_base_path: str | Path,
    weighting: str,
    head: int | None,
    log: BoundLogger,
    stop_event: threading.Event,
) -> int:
    """Build Term-Document Matrix from corpus fragments."""
    output_base_path = Path(output_base_path)
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
        fragments_path = doc_store.get_fragments_by_build_id(corpus_build_id)
        log.info("fragments queried", extra={"count": len(fragments_path)})

        if head is not None:
            fragments_path = fragments_path[:head]
            log.info("limited to head", extra={"head": head, "count": len(fragments_path)})

        build_id = doc_store.insert_tdm_build(
            corpus_id=corpus_build_id,
            weighting=weighting,
            status=ProcessingStatus.RUNNING,
        )
        log.info("tdm build created", extra={"build_id": build_id})

        tf_matrix_rows: list[int] = []
        tf_matrix_cols: list[int] = []
        tf_matrix_vals: list[int] = []
        max_token_id = 0

        row_idx_mapping: list[tuple[int, str]] = []
        for row_idx, fragment in enumerate(
            iter_fragments(storage, fragments_path, stop_event, log)
        ):
            row_idx_mapping.append((row_idx, fragment.id))
            doc_tf = Counter(fragment.token_ids)
            for token_id, count in doc_tf.items():
                tf_matrix_rows.append(row_idx)
                tf_matrix_cols.append(token_id)
                tf_matrix_vals.append(count)
                max_token_id = max(max_token_id, token_id)

        n_fragments = len(row_idx_mapping)
        log.info("all fragments loaded", extra={"n_fragments": n_fragments})

        if n_fragments == 0:
            log.warning("no fragments loaded, nothing to build")
            doc_store.update_tdm_build(
                build_id, status=ProcessingStatus.OK, stats={"n_fragments": 0}
            )
            return 0

        tf_matrix: sparse.csr_matrix = sparse.csr_matrix(  # type: ignore[arg-type]
            (tf_matrix_vals, (tf_matrix_rows, tf_matrix_cols)),
            shape=(n_fragments, max_token_id + 1),
        )

        weighting_fn = WEIGHTING_METHODS_MAPPING[weighting]
        result_matrix = weighting_fn(tf_matrix)

        temp_npz_file = storage.create_temp_file(".tdm.npz")
        temp_tsv_file = storage.create_temp_file(".tdm.tsv")

        try:
            sparse.save_npz(temp_npz_file, result_matrix)
            with open(temp_tsv_file, "w", encoding="utf-8") as f:
                f.write("row_idx\tfragment_id\n")
                for idx, fragment_id in row_idx_mapping:
                    f.write(f"{idx}\t{fragment_id}\n")

            storage.finalize_artifact(
                temp_npz_file, output_base_path / "matrix.npz", blocking=True
            )
            storage.finalize_artifact(
                temp_tsv_file, output_base_path / "row_index.tsv", blocking=True
            )
        except Exception:
            try:
                temp_npz_file.unlink(missing_ok=True)
                temp_tsv_file.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        stats = {
            "n_fragments": n_fragments,
            "vocab_size": result_matrix.shape[1],
            "nnz": int(result_matrix.nnz),
            "matrix_shape_rows": result_matrix.shape[0],
            "matrix_shape_cols": result_matrix.shape[1],
        }
        doc_store.update_tdm_build(
            build_id,
            status=ProcessingStatus.OK,
            path=str(output_base_path),
            stats=stats,
        )
        log.info("tdm build completed", extra={"build_id": build_id, **stats})

    except Exception:
        if build_id is not None:
            try:
                doc_store.update_tdm_build(build_id, status=ProcessingStatus.ERROR)
            except Exception:
                pass
        raise
    finally:
        storage.close()
        doc_store.close()
        log.info("resources closed")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for build_tdm pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build Term-Document Matrix from corpus fragments."
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
        "--corpus-build-id",
        type=int,
        required=True,
        help="run_id from corpus_builds",
    )
    parser.add_argument(
        "--output-base-path",
        "-output",
        type=str,
        required=True,
        help="Base directory for TDM output (relative to data-path)",
    )
    parser.add_argument(
        "--weighting-method",
        type=str,
        choices=["tf-idf", "log-entropy"],
        default="log-entropy",
        help="Weighting method (default: log-entropy)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Limit number of fragments (for testing)",
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

    log_file = Path(args.log_file or data_path / "logs/embeddings/build_tdm.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(fmt=args.log_format, level=args.log_level, file=str(log_file))
    log = get_logger("root").bind(pipeline="build_tdm")

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
            corpus_build_id=args.corpus_build_id,
            output_base_path=args.output_base_path,
            weighting=args.weighting_method,
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
