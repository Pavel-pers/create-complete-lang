#!/usr/bin/env python3
"""
CLI: Build clustering from word embeddings.

Usage:
    python build_clustering.py --method wishart --embeddings-version v1 --version c1 --target-clusters 18
    python build_clustering.py --method wishart --embeddings-version v1 --version c2 --knn-k 20 --target-clusters 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from storage_setup import make_doc_store, EXPERIMENT_DATA
from vocab_builder import load_vocab
from cclang.io.schemas import ProcessingStatus


CLUSTERING_METHODS = {}


def register_method(name: str):
    def decorator(fn):
        CLUSTERING_METHODS[name] = fn
        return fn
    return decorator


@register_method("wishart")
def _build_wishart(vectors, args):
    from clustering.wishart import build
    return build(
        vectors=vectors,
        knn_k=args.knn_k,
        target_clusters=args.target_clusters,
        r_cut=args.r_cut,
        min_cluster_size=args.min_cluster_size,
        merge_small=not args.no_merge_small,
        assign_noise=not args.no_assign_noise,
    )


@register_method("kmeans")
def _build_kmeans(vectors, args):
    from clustering.kmeans_method import build
    n = args.target_clusters or 18
    return build(
        vectors=vectors,
        n_clusters=n,
        random_state=42,
    )


@register_method("bisecting-kmeans")
def _build_bisecting(vectors, args):
    from clustering.bisecting_kmeans import build
    n = args.target_clusters or 18
    return build(
        vectors=vectors,
        n_clusters=n,
        random_state=42,
        min_cluster_size=args.min_cluster_size,
    )


@register_method("dbscan")
def _build_dbscan(vectors, args):
    from clustering.density_methods import build_dbscan
    return build_dbscan(
        vectors=vectors,
        eps=args.eps if args.eps else 0.2,
        min_samples=args.knn_k,
        assign_noise=not args.no_assign_noise,
    )


@register_method("hdbscan")
def _build_hdbscan(vectors, args):
    from clustering.density_methods import build_hdbscan
    return build_hdbscan(
        vectors=vectors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.knn_k,
        cluster_selection_method="eom",
        assign_noise=not args.no_assign_noise,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build clustering from word embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # General
    parser.add_argument("--method", required=True, choices=list(CLUSTERING_METHODS.keys()))
    parser.add_argument("--embeddings-version", required=True,
                        help="Which embeddings to use (e.g. v1)")
    parser.add_argument("--version", required=True,
                        help="Clustering artifact version (e.g. c1)")

    # Wishart / DBSCAN / HDBSCAN params
    parser.add_argument("--knn-k", type=int, default=15, help="KNN k / DBSCAN min_samples")
    parser.add_argument("--target-clusters", type=int, default=None)
    parser.add_argument("--r-cut", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps")
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--no-merge-small", action="store_true")
    parser.add_argument("--no-assign-noise", action="store_true")

    # Workflow
    parser.add_argument("--no-db", action="store_true")

    args = parser.parse_args(argv)

    emb_dir = EXPERIMENT_DATA / "embeddings" / args.embeddings_version
    out_dir = EXPERIMENT_DATA / "clusters" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Clustering: method={args.method}, version={args.version}")
    print(f"Embeddings: {emb_dir}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # ── Load embeddings ──
    vectors = np.load(emb_dir / "word_vectors.npy")
    print(f"[load] vectors shape: {vectors.shape}")

    # ── Load vocab for reference ──
    vocab_path = emb_dir / "vocab.tsv"
    if vocab_path.exists():
        idx2word, _, _, _ = load_vocab(vocab_path)
    else:
        idx2word = [f"word_{i}" for i in range(vectors.shape[0])]

    # ── DB registration ──
    build_id = None
    if not args.no_db:
        try:
            doc_store = make_doc_store()
            build_id = doc_store.register_cluster_build(
                path=str(out_dir),
                version=args.version,
                status=ProcessingStatus.RUNNING,
            )
            print(f"[db] registered cluster_build run_id={build_id}")
        except Exception as e:
            print(f"[db] WARNING: could not register build: {e}")
            build_id = None

    try:
        # ── Build clustering ──
        build_fn = CLUSTERING_METHODS[args.method]
        labels, densities, meta = build_fn(vectors, args)

        # ── Save labels ──
        np.savez_compressed(
            out_dir / "labels.npz",
            cluster_ids=labels.astype(np.int32),
            densities=densities.astype(np.float32),
            roles=np.zeros(len(labels), dtype=np.int8),
        )

        # ── Save hierarchy if available (Wishart) ──
        hierarchy = meta.pop("_hierarchy", None)
        if hierarchy is not None:
            np.savez_compressed(
                out_dir / "hierarchy.npz",
                event_radii=hierarchy["event_radii"],
                event_types=hierarchy["event_types"],
                event_payload=hierarchy["event_payload"],
                knn_radii=hierarchy["knn_radii"],
            )

        # ── Save manifest ──
        manifest = {
            "schema_version": "0.2.0",
            "artifact_type": "cluster",
            "method": args.method,
            "version": args.version,
            "params": {
                "embeddings_version": args.embeddings_version,
                **{k: v for k, v in meta.items()},
            },
            "created_at": datetime.now().isoformat() + "Z",
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

        print(f"\nArtifacts saved: {out_dir}")

        # ── Complete DB ──
        if build_id is not None:
            try:
                doc_store.complete_cluster_build(build_id, ProcessingStatus.OK)
                print(f"[db] completed cluster_build run_id={build_id}")
            except Exception as e:
                print(f"[db] WARNING: could not complete build: {e}")

    except Exception:
        if build_id is not None:
            try:
                doc_store.complete_cluster_build(build_id, ProcessingStatus.ERROR)
            except Exception:
                pass
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
