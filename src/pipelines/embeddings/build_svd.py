from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from dotenv import load_dotenv
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from cclang.common.logx import BoundLogger, get_logger, setup_logging
from cclang.config.s3 import load_s3_config
from cclang.core.storage import CloudConfig, StorageManager
from cclang.io.db import get_conn
from cclang.io.doc_state_store import DocStateStore
from cclang.io.fs import LocalConfig
from cclang.io.schemas import (
    ProcessingStatus, SvdInputInfo, SvdParams, SvdSpectrum, SvdSigmaSummary, SvdRuntime, SvdBuildStats)


def run_pipeline(
        data_path: Path,
        database_dsn: str | None,
        tdm_build_id: int,
        k: int,
        n_iter: int,
        oversampling: int,
        random_state: int | None,
        output_base_path: str | Path,
        log: BoundLogger,
        stop_event: threading.Event,
) -> int:
    """Compute truncated SVD on a weighted Term-Document Matrix."""

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
        # --- Resolve TDM path from DB ---
        tdm_info = doc_store.get_tdm_build(tdm_build_id)
        if tdm_info.status != ProcessingStatus.OK:
            log.error(f"tdm_build_id {tdm_build_id} is not successful")
            return 1

        tdm_base_path = tdm_info.path
        tdm_path = Path(tdm_base_path) / 'matrix.npz'

        log.info('start matrix load')
        with storage.open(tdm_path, mode='rb') as tdm_file:
            tdm = sparse.load_npz(tdm_file)

        n_docs, n_tokens = tdm.shape
        log.info(f"finished loading tdm", extra={'n_docs': n_docs, 'n_tokens': n_tokens})

        if stop_event.is_set():
            log.warning('stop requested')
            return 130

        params = {
            "k": k,
            "n_iter": n_iter,
            "oversampling": oversampling,
            "random_state": random_state,
        }
        build_id = doc_store.insert_svd_build(
            tdm_id=tdm_build_id,
            k=k,
            params=params,
            status=ProcessingStatus.RUNNING,
        )

        log.info("starting randomized SVD", extra={"k": k, "n_iter": n_iter})

        t0 = time.monotonic()

        U, sigma, Vt = randomized_svd(
            tdm,
            n_components=k,
            n_iter=n_iter,
            n_oversamples=oversampling,
            random_state=random_state,
        )

        wall_time = time.monotonic() - t0
        log.info(
            "SVD computed",
            extra={
                "wall_time_seconds": round(wall_time, 2),
                "U_shape": list(U.shape),
                "sigma_shape": list(sigma.shape),
                "Vt_shape": list(Vt.shape),
            },
        )

        if stop_event.is_set():
            log.warning('stop requested, ignoring')

        # --- Compute stats ---
        forb_norm = float(np.sum(tdm.data ** 2))
        forb_explained_ratio = sigma ** 2 / forb_norm
        forb_explained_cumulative = np.cumsum(forb_explained_ratio)
        forb_total_explained = forb_explained_cumulative[-1]

        cumulative_arr = np.array(forb_explained_cumulative)
        effective_rank_90 = int(np.searchsorted(cumulative_arr, 0.90) + 1) if len(cumulative_arr) else k
        effective_rank_95 = int(np.searchsorted(cumulative_arr, 0.95) + 1) if len(cumulative_arr) else k

        # --- Create stats object
        svd_input_stats = SvdInputInfo(
            tdm_build_id=tdm_build_id,
            matrix_shape=[n_docs, n_tokens],
            matrix_nnz=tdm.nnz,
            matrix_density=tdm.nnz / (n_docs * n_tokens),
        )
        svd_params = SvdParams(
            k=k,
            n_iter=n_iter,
            oversampling=oversampling,
            random_state=random_state,
        )
        svd_spectrum_stats = SvdSpectrum(
            singular_values=np.round(sigma, decimals=1).tolist(),
            explained_variance_ratio=np.round(forb_explained_ratio, decimals=2).tolist(),
            cumulative_energy=np.round(forb_explained_cumulative, decimals=2).tolist(),
            energy_captured=round(forb_total_explained, 3),
            effective_rank_90=effective_rank_90,
            effective_rank_95=effective_rank_95
        )
        svd_sigma_stats = SvdSigmaSummary(
            max=round(np.max(sigma), 2),
            min=round(np.min(sigma), 2),
            median=round(np.median(sigma), 2),
            mean=round(np.mean(sigma), 3),
        )
        svd_runtime_info_stats = SvdRuntime(
            wall_time_seconds=round(wall_time, 1)
        )
        svd_full_stats = SvdBuildStats(
            input=svd_input_stats,
            params=svd_params,
            spectrum=svd_spectrum_stats,
            sigma_summary=svd_sigma_stats,
            runtime=svd_runtime_info_stats
        )


        # --- Save artifacts ---

        temp_u = storage.create_temp_file(".U.npy")
        temp_sigma = storage.create_temp_file(".sigma.npy")
        temp_vt = storage.create_temp_file(".Vt.npy")
        temp_json_stats = storage.create_temp_file(".stats.json")
        try:
            np.save(temp_u, U)
            np.save(temp_sigma, sigma)
            np.save(temp_vt, Vt)
            with open(temp_json_stats, 'w', encoding='utf-8') as f:
                json.dump(svd_full_stats.model_dump(mode='json'), f, ensure_ascii=False, indent=4)

            storage.finalize_artifact(
                temp_u, output_base_path / "U.npy", blocking=True
            )
            storage.finalize_artifact(
                temp_sigma, output_base_path / "sigma.npy", blocking=True
            )
            storage.finalize_artifact(
                temp_vt, output_base_path / "Vt.npy", blocking=True
            )
            storage.finalize_artifact(
                temp_json_stats, output_base_path / "stats.json", blocking=True
            )
        except Exception:
            try:
                temp_u.unlink(missing_ok=True)
                temp_sigma.unlink(missing_ok=True)
                temp_vt.unlink(missing_ok=True)
                temp_json_stats.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        stats_highlight = {
            "matrix_shape": [n_docs, n_tokens],
            "k": k,
            "forb_explained": forb_total_explained,
            "effective_rank_90": effective_rank_90,
            "effective_rank_95": effective_rank_95,
            "wall_time_seconds": round(wall_time, 2),
        }
        doc_store.update_svd_build(
            build_id,
            status=ProcessingStatus.OK,
            path=str(output_base_path),
            stats=stats_highlight,
        )
        log.info(
            "svd build completed",
            extra={
                "build_id": build_id,
                "k": k,
                "wall_time_seconds": stats_highlight["wall_time_seconds"],
            },
        )

    except Exception:
        if build_id is not None:
            try:
                doc_store.update_svd_build(build_id, status=ProcessingStatus.ERROR)
            except Exception:
                pass
        raise
    finally:
        storage.close()
        doc_store.close()
        log.info("resources closed")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for build_svd pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compute truncated SVD on a Term-Document Matrix."
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
        "--tdm-build-id",
        type=int,
        required=True,
        help="run_id from tdm_builds",
    )
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Number of singular values/vectors to compute (rank of truncation)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="Number of power iterations for randomized SVD (default: 5)",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=10,
        help="Oversampling parameter p for randomized SVD (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--output-base-path",
        "-output",
        type=str,
        required=True,
        help="Base directory for SVD output (relative to data-path)",
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

    log_file = Path(args.log_file or data_path / "logs/embeddings/build_svd.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(fmt=args.log_format, level=args.log_level, file=str(log_file))
    log = get_logger("root").bind(pipeline="build_svd")

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
            tdm_build_id=args.tdm_build_id,
            k=args.k,
            n_iter=args.n_iter,
            oversampling=args.oversampling,
            random_state=args.random_state,
            output_base_path=args.output_base_path,
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
