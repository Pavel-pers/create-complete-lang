"""Pipeline: build word embeddings from SVD decomposition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from dotenv import load_dotenv

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.fs import LocalConfig
from cclang.io.schemas import ProcessingStatus


def run_pipeline(
    data_path: Path,
    database_dsn: str | None,
    svd_build_id: int,
    sigma_power: float,
    reshape_k: int | None,
    output_base_path: str | Path,
    log: BoundLogger,
) -> int:
    """Build word embeddings from SVD: word_vectors = Vt.T · diag(sigma^alpha)."""

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
        # --- Resolve SVD path from DB ---
        svd_info = doc_store.get_svd_build(svd_build_id)
        if svd_info.status != ProcessingStatus.OK:
            log.error(
                "svd build is not OK",
                extra={"svd_build_id": svd_build_id, "status": svd_info.status.value},
            )
            return 1

        svd_path = svd_info.path
        if not svd_path:
            log.error("svd build has no path", extra={"svd_build_id": svd_build_id})
            return 1

        # --- Load sigma and Vt ---
        sigma_file = Path(svd_path) / "sigma.npy"
        vt_file = Path(svd_path) / "Vt.npy"

        log.info("loading SVD artifacts", extra={"svd_path": svd_path})

        with storage.open(sigma_file, mode="rb") as f:
            sigma = np.load(f)
        with storage.open(vt_file, mode="rb") as f:
            Vt = np.load(f)

        k = len(sigma)
        if reshape_k is None:
            reshape_k = k

        vocab_size = Vt.shape[1]
        log.info(
            "SVD artifacts loaded",
            extra={"k": k, "vocab_size": vocab_size, "sigma_power": sigma_power},
        )

        # --- Register build in DB ---
        build_id = doc_store.insert_embedding_build(
            svd_id=svd_build_id,
            sigma_power=sigma_power,
            reshape_k=reshape_k,
            method="svd",
            status=ProcessingStatus.RUNNING,
        )
        log.info("embedding build created", extra={"build_id": build_id})

        # --- Compute word vectors ---
        #  Vt shape: (k, vocab_size)
        #  word_vectors shape: (vocab_size, k)
        if sigma_power == 0.0:
            word_vectors = Vt.T[:, :reshape_k]
        elif sigma_power == 1.0:
            word_vectors = Vt.T[:, :reshape_k] * sigma[np.newaxis, :reshape_k]
        else:
            word_vectors = Vt.T[:, :reshape_k] * (sigma ** sigma_power)[np.newaxis, :reshape_k]

        # --- Compute stats ---
        norms = np.linalg.norm(word_vectors, axis=1)
        stats = {
            "vocab_size": vocab_size,
            "k": reshape_k,
            "sigma_power": sigma_power,
            "vector_norm_mean": round(float(np.mean(norms)), 4),
            "vector_norm_std": round(float(np.std(norms)), 4),
            "vector_norm_min": round(float(np.min(norms)), 6),
            "vector_norm_max": round(float(np.max(norms)), 4),
            "zero_norm_count": int(np.sum(norms == 0.0)),
        }
        log.info("word vectors computed", extra=stats)

        # --- Save artifacts ---
        temp_vectors = storage.create_temp_file(".word_vectors.npy")
        temp_meta = storage.create_temp_file(".meta.json")

        try:
            np.save(temp_vectors, word_vectors)

            meta = {
                "schema_version": "0.1.0",
                "svd_build_id": svd_build_id,
                "method": "svd",
                "k": reshape_k,
                "sigma_power": sigma_power,
                "vocab_size": vocab_size,
                "stats": stats,
            }
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            storage.finalize_artifact(
                temp_vectors, output_base_path / "word_vectors.npy", blocking=True
            )
            storage.finalize_artifact(
                temp_meta, output_base_path / "meta.json", blocking=True
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
            build_id,
            status=ProcessingStatus.OK,
            path=str(output_base_path),
            stats=stats,
        )
        log.info(
            "embedding build completed",
            extra={"build_id": build_id, **stats},
        )

    except Exception:
        if build_id is not None:
            try:
                doc_store.update_embedding_build(build_id, status=ProcessingStatus.ERROR)
            except Exception:
                pass
        raise
    finally:
        storage.close()
        doc_store.close()
        log.info("resources closed")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for build_embeddings pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build word embeddings from SVD decomposition (V·Σ^α)."
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
        "--svd-build-id",
        type=int,
        required=True,
        help="run_id from svd_builds",
    )
    parser.add_argument(
        "--sigma-power",
        type=float,
        default=1.0,
        help="Exponent α for V·Σ^α (0=unweighted, 0.5=sqrt, 1=full). Default: 1.0",
    )
    parser.add_argument(
        "-k",
        "--reshape-k",
        type=int,
        default=None,
        help="Number of dimensions to keep (default: all k from SVD)",
    )
    parser.add_argument(
        "--output-base-path",
        "-output",
        type=str,
        required=True,
        help="Base directory for embeddings output (relative to data-path)",
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

    log_file = Path(args.log_file or data_path / "logs/embeddings/build_embeddings.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(fmt=args.log_format, level=args.log_level, file=str(log_file))
    log = get_logger("root").bind(pipeline="build_embeddings")

    if not database_dsn:
        log.error("database DSN not provided. Use --database-dsn or set CCLANG_DB_DSN")
        return 1

    try:
        return run_pipeline(
            data_path=data_path,
            database_dsn=database_dsn,
            svd_build_id=args.svd_build_id,
            sigma_power=args.sigma_power,
            reshape_k=args.reshape_k,
            output_base_path=args.output_base_path,
            log=log,
        )
    except Exception:
        log.exception("unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())