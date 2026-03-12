"""Post-processing pipeline for SVD embeddings.

Applies a configurable chain of transformations to raw V*Sigma^alpha embeddings.
Each step is optional and controlled via CLI flags.

Available steps (applied in this order):
  1. center     -- subtract the mean vector (zero-centering).
  2. pca-remove -- remove projections onto top-D principal components
                   ("all-but-the-top", Mu & Viswanath 2018).
  3. normalize  -- L2-normalize each row to unit length.

Recommended configurations:

  --steps normalize
      Minimal fix.  Removes frequency-norm coupling.

  --steps center,normalize
      Zero-centering before normalization.

  --steps center,pca-remove,normalize  --pca-d 7
      Full Mu & Viswanath (2018).  Removes top-D principal components.
      Test D in {1, 3, 5, 7, 10}.

Usage
-----
    python postprocess_embeddings.py \\
        --build-id 5 \\
        --steps center,pca-remove,normalize \\
        --pca-d 7

    # Compare multiple D values:
    for d in 1 3 5 7 10; do
        python postprocess_embeddings.py \\
            --build-id 5 \\
            --steps center,pca-remove,normalize \\
            --pca-d $d
    done

References
----------
- Mu & Viswanath (2018) "All-but-the-Top: Simple and Effective
  Postprocessing for Word Representations".  ICLR 2018.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.preprocessing import normalize

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.fs import LocalConfig
from cclang.io.schemas import ProcessingStatus


# ======================================================================
# Individual transforms
# ======================================================================

def step_center(X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract mean vector from all rows."""
    mean_vec = X.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    X_centered = X - mean_vec
    return X_centered, {
        "step": "center",
        "mean_vector_norm": round(mean_norm, 6),
    }


def step_pca_remove(
        X: np.ndarray,
        d: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove projections onto top-D principal components.

    "All-but-the-top" (Mu & Viswanath 2018):
      1. Compute covariance of (already centred) X.
      2. Find top-D eigenvectors.
      3. Subtract their projections: X' = X - X @ P @ P^T
    """
    cov = np.cov(X.T)                            # (k, k)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order; reverse to get top-D
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    P = eigenvectors[:, :d]                       # (k, d)

    total_var = float(eigenvalues.sum())
    removed_var = float(eigenvalues[:d].sum())

    projections = X @ P                           # (n, d)
    X_cleaned = X - projections @ P.T             # (n, k)

    return X_cleaned, {
        "step": "pca_remove",
        "d": d,
        "variance_removed_abs": round(removed_var, 4),
        "variance_removed_pct": round(removed_var / total_var * 100, 2),
        "top_eigenvalues": [round(float(v), 4) for v in eigenvalues[:d]],
    }


def step_normalize(X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """L2-normalize each row to unit length."""
    norms_before = np.linalg.norm(X, axis=1)

    X_normed = normalize(X, axis=1, norm="l2")

    return X_normed, {
        "step": "normalize",
        "norms_before_mean": round(float(norms_before.mean()), 6),
        "norms_before_std": round(float(norms_before.std()), 6),
        "norms_before_min": round(float(norms_before.min()), 6),
        "norms_before_max": round(float(norms_before.max()), 6),
    }


# ======================================================================
# Pipeline core
# ======================================================================

VALID_STEPS = ["center", "pca-remove", "normalize"]


def run_postprocess(
        X: np.ndarray,
        steps: list[str],
        pca_d: int = 7,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply a chain of transformations, returning the result and a log."""
    log: list[dict[str, Any]] = []

    for step_name in steps:
        if step_name == "center":
            X, info = step_center(X)
        elif step_name == "pca-remove":
            X, info = step_pca_remove(X, d=pca_d)
        elif step_name == "normalize":
            X, info = step_normalize(X)
        else:
            raise ValueError(f"Unknown step: {step_name!r}.  Valid: {VALID_STEPS}")
        log.append(info)

    return X, log


# ======================================================================
# DB-integrated pipeline
# ======================================================================

def run_pipeline(
    data_path: Path,
    database_dsn: str | None,
    build_id: int,
    steps: list[str],
    pca_d: int,
    output_base_path: str | Path | None,
    log: BoundLogger,
) -> int:
    """Post-process embeddings identified by *build_id*.

    1. Fetch the source ``embedding_builds`` row.
    2. Load ``word_vectors.npy`` from its artifact path.
    3. Create a **new** ``embedding_builds`` row inheriting
       ``svd_id``, ``sigma_power``, ``reshape_k``, ``method``, ``stats``
       from the source and storing the postprocess config in ``params``.
    4. Apply the requested transformation chain.
    5. Save artifacts and update the new row with ``status=ok``.
    """

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

    new_build_id: int | None = None

    try:
        # --- Resolve source embedding build ---
        source = doc_store.get_embeddings_build(build_id)
        if source.status != ProcessingStatus.OK:
            log.error(
                "source embedding build is not OK",
                extra={"build_id": build_id, "status": source.status.value},
            )
            return 1

        if not source.path:
            log.error("source embedding build has no path", extra={"build_id": build_id})
            return 1

        log.info(
            "source build resolved",
            extra={
                "build_id": build_id,
                "svd_id": source.svd_id,
                "sigma_power": source.sigma_power,
                "reshape_k": source.reshape_k,
                "method": source.method,
                "path": source.path,
            },
        )

        # --- Load word_vectors.npy ---
        vectors_file = Path(source.path) / "word_vectors.npy"
        log.info("loading embeddings", extra={"vectors_file": str(vectors_file)})

        with storage.open(vectors_file, mode="rb") as f:
            X = np.load(f).astype(np.float64)

        log.info("embeddings loaded", extra={"shape": list(X.shape)})

        # --- Build params for the new row ---
        params: dict[str, Any] = {
            "source_build_id": build_id,
            "steps": steps,
            "pca_d": pca_d if "pca-remove" in steps else None,
        }

        # --- Register new build in DB (inheriting from source) ---
        new_build_id = doc_store.insert_embedding_build(
            svd_id=source.svd_id,
            sigma_power=source.sigma_power,
            reshape_k=source.reshape_k,
            method=source.method,
            status=ProcessingStatus.RUNNING,
            params=params,
            stats=source.stats,
        )
        log.info("new postprocess build created", extra={"new_build_id": new_build_id})

        # --- Determine output path ---
        if output_base_path is None:
            output_base_path = Path(
                f"artifacts/postprocessed_embeddings/build_{new_build_id}"
            )
        else:
            output_base_path = Path(output_base_path)

        # --- Run postprocessing ---
        t0 = time.time()
        log.info("applying steps", extra={"steps": steps, "pca_d": pca_d})
        X_out, step_log = run_postprocess(X, steps, pca_d=pca_d)
        elapsed = time.time() - t0

        for entry in step_log:
            log.info("step completed", extra=entry)

        # --- Update params with results ---
        params["step_details"] = step_log
        params["elapsed_sec"] = round(elapsed, 2)
        params["input_shape"] = list(X.shape)
        params["output_shape"] = list(X_out.shape)

        # --- Save artifacts ---
        temp_vectors = storage.create_temp_file(".word_vectors.npy")
        temp_meta = storage.create_temp_file(".meta.json")

        try:
            np.save(temp_vectors, X_out.astype(np.float32))

            meta = {
                "schema_version": "0.1.0",
                "source_build_id": build_id,
                "new_build_id": new_build_id,
                "svd_id": source.svd_id,
                "method": source.method,
                "sigma_power": source.sigma_power,
                "reshape_k": source.reshape_k,
                "steps": steps,
                "pca_d": pca_d if "pca-remove" in steps else None,
                "shape": list(X_out.shape),
                "elapsed_sec": round(elapsed, 2),
                "step_details": step_log,
            }
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            storage.finalize_artifact(
                temp_vectors, output_base_path / "word_vectors.npy", blocking=True,
            )
            storage.finalize_artifact(
                temp_meta, output_base_path / "meta.json", blocking=True,
            )
        except Exception:
            for tmp in (temp_vectors, temp_meta):
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        # --- Update DB ---
        doc_store.update_embedding_build(
            new_build_id,
            status=ProcessingStatus.OK,
            path=str(output_base_path),
            params=params,
        )
        log.info(
            "postprocess build completed",
            extra={
                "new_build_id": new_build_id,
                "source_build_id": build_id,
                "steps": steps,
                "elapsed_sec": round(elapsed, 2),
            },
        )

    except Exception:
        if new_build_id is not None:
            try:
                doc_store.update_embedding_build(
                    new_build_id, status=ProcessingStatus.ERROR,
                )
            except Exception:
                pass
        raise
    finally:
        storage.close()
        doc_store.close()
        log.info("resources closed")

    return 0


# ======================================================================
# CLI
# ======================================================================

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for postprocess_embeddings pipeline."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Post-process SVD embeddings: center, PCA-remove, L2-normalize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --build-id 5 --steps normalize\n"
            "  %(prog)s --build-id 5 --steps center,pca-remove,normalize --pca-d 7\n"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Root data directory (env: CCLANG_DATA_DIR, default: data)",
    )
    parser.add_argument(
        "--database-dsn", "-db",
        type=str,
        default=None,
        help="PostgreSQL DSN (or env CCLANG_DB_DSN)",
    )
    parser.add_argument(
        "--build-id",
        type=int,
        required=True,
        help="run_id of the source embedding_builds row to post-process",
    )
    parser.add_argument(
        "--steps", "-s",
        type=str,
        required=True,
        help=(
            "Comma-separated processing steps in order.  "
            f"Available: {', '.join(VALID_STEPS)}.  "
            "Example: center,pca-remove,normalize"
        ),
    )
    parser.add_argument(
        "--pca-d",
        type=int,
        default=7,
        help="Number of top PCs to remove in pca-remove step (default: 7)",
    )
    parser.add_argument(
        "--output-base-path", "-output",
        type=str,
        default=None,
        help=(
            "Base directory for output artifacts (relative to data-path). "
            "Default: artifacts/postprocessed_embeddings/build_<new_id>"
        ),
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

    # Validate steps early
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in steps:
        if s not in VALID_STEPS:
            print(f"ERROR: unknown step {s!r}.  Valid: {VALID_STEPS}", file=sys.stderr)
            return 1

    data_path = args.data_path or Path(os.environ.get("CCLANG_DATA_DIR", "data"))
    database_dsn = args.database_dsn or os.environ.get("CCLANG_DB_DSN")

    log_file = Path(
        args.log_file or data_path / "logs/embeddings/postprocess_embeddings.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(fmt=args.log_format, level=args.log_level, file=str(log_file))
    log = get_logger("root").bind(pipeline="postprocess_embeddings")

    if not database_dsn:
        log.error("database DSN not provided. Use --database-dsn or set CCLANG_DB_DSN")
        return 1

    try:
        return run_pipeline(
            data_path=data_path,
            database_dsn=database_dsn,
            build_id=args.build_id,
            steps=steps,
            pca_d=args.pca_d,
            output_base_path=args.output_base_path,
            log=log,
        )
    except Exception:
        log.exception("unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
