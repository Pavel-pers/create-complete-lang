#!/usr/bin/env python3
"""
Reduce embedding dimensionality via SVD.

Usage:
    python svd_reduce.py --source v9-cbow300 --target v9-cbow300-svd50 --target-dim 50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.utils.extmath import randomized_svd


def reduce_svd(source: str, target: str, target_dim: int = 50) -> None:
    root = Path(__file__).resolve().parent / "data" / "embeddings"
    src_dir = root / source
    tgt_dir = root / target

    wv = np.load(src_dir / "word_vectors.npy")
    print(f"Source: {src_dir} shape={wv.shape}")

    # Center before SVD (for PCA-like reduction)
    mean = wv.mean(axis=0)
    centered = wv - mean

    # SVD: U Σ V^T where U is (n, dim), Σ (dim,), V (dim, feat)
    U, s, Vt = randomized_svd(centered, n_components=target_dim, n_iter=5, random_state=42)
    print(f"Top singular values (first 10): {s[:10]}")
    print(f"Cumulative variance captured: {(s**2).cumsum()[-1] / (centered**2).sum():.4f}")

    # Reduced embeddings = U * Σ (word_vectors in reduced space)
    reduced = U * s

    # L2 normalize
    normed = sk_normalize(reduced, norm="l2", axis=1)

    tgt_dir.mkdir(parents=True, exist_ok=True)
    np.save(tgt_dir / "word_vectors.npy", normed)
    shutil.copy(src_dir / "vocab.tsv", tgt_dir / "vocab.tsv")

    meta = {
        "schema_version": "0.2.0",
        "artifact_type": "embedding",
        "method": "svd-reduction",
        "version": target,
        "params": {
            "base_version": source,
            "target_dim": target_dim,
            "postprocess": ["center", "svd_reduce", "l2_normalize"],
            "top_singular_values": s[:10].tolist(),
        },
    }
    with open(tgt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Saved: {tgt_dir}, shape={normed.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-dim", type=int, default=50)
    args = parser.parse_args()
    reduce_svd(args.source, args.target, args.target_dim)


if __name__ == "__main__":
    main()
