#!/usr/bin/env python3
"""
Run Bisecting K-Means at 3 granularity levels on an embedding version.

Usage:
    python run_three_levels.py --embeddings-version v7-svd50-nopc1 --prefix c_p2_v7pc1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from clustering.bisecting_kmeans import build as bisecting_build
from vocab_builder import load_vocab


def run_level(
    vectors: np.ndarray,
    embeddings_version: str,
    level_name: str,
    target_clusters: int,
    min_cluster_size: int,
    cluster_version: str,
    out_root: Path,
) -> dict:
    """Run one clustering level and save results."""
    out_dir = out_root / cluster_version
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, densities, meta = bisecting_build(
        vectors=vectors,
        n_clusters=target_clusters,
        random_state=42,
        min_cluster_size=min_cluster_size,
    )

    np.savez_compressed(
        out_dir / "labels.npz",
        cluster_ids=labels.astype(np.int32),
        densities=densities.astype(np.float32),
        roles=np.zeros(len(labels), dtype=np.int8),
    )

    # Manifest
    manifest = {
        "schema_version": "0.2.0",
        "artifact_type": "cluster",
        "method": "bisecting-kmeans",
        "version": cluster_version,
        "params": {
            "embeddings_version": embeddings_version,
            "level": level_name,
            "target_clusters": target_clusters,
            "min_cluster_size": min_cluster_size,
            **{k: v for k, v in meta.items() if not k.startswith("_")},
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Summary
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    summary = {
        "cluster_version": cluster_version,
        "embeddings_version": embeddings_version,
        "level": level_name,
        "n_clusters": int(len(unique)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": round(float(counts.mean()), 1),
        "std": round(float(counts.std()), 1),
        "cv": round(float(counts.std() / counts.mean()), 4),
        "max_min_ratio": round(float(counts.max() / counts.min()), 2),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-version", required=True)
    parser.add_argument("--prefix", required=True, help="Cluster version prefix (e.g. c_p2_v7pc1)")
    parser.add_argument("--min-cluster-size", type=int, default=95)
    args = parser.parse_args()

    emb_dir = HERE / "data" / "embeddings" / args.embeddings_version
    out_root = HERE / "data" / "clusters"

    vectors = np.load(emb_dir / "word_vectors.npy")
    print(f"[load] {args.embeddings_version}: shape={vectors.shape}")

    levels = [
        ("large", 18, f"{args.prefix}-large"),
        ("medium", 100, f"{args.prefix}-medium"),
        ("small", 800, f"{args.prefix}-small"),
    ]

    results = []
    for name, target, version in levels:
        print(f"\n=== {name} (target={target}) ===")
        s = run_level(vectors, args.embeddings_version, name, target, args.min_cluster_size,
                      version, out_root)
        results.append(s)
        print(f"  {s}")

    print("\n=== Summary ===")
    for s in results:
        print(f"  {s['cluster_version']:30s} k={s['n_clusters']:4d} "
              f"min={s['min']:4d} max={s['max']:4d} cv={s['cv']:.3f} ratio={s['max_min_ratio']:.2f}")

    # Save combined summary
    summary_path = out_root / f"{args.prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
