#!/usr/bin/env python3
"""
Create a nopc1 version of embeddings by removing the first principal component.

Usage:
    python make_nopc1.py --source v7-svd50 --target v7-svd50-nopc1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.utils.extmath import randomized_svd


def make_nopc1(source: str, target: str) -> None:
    root = Path(__file__).resolve().parent / "data" / "embeddings"
    src_dir = root / source
    tgt_dir = root / target

    wv = np.load(src_dir / "word_vectors.npy")
    print(f"Source: {src_dir} shape={wv.shape}")

    # Center and find PC1
    centered = wv - wv.mean(axis=0)
    U, s, Vt = randomized_svd(centered, n_components=5, n_iter=3, random_state=42)
    print(f"Top 5 singular values: {s}")

    # Remove PC1
    pc1 = Vt[0]
    projection = centered @ pc1
    cleaned = centered - np.outer(projection, pc1)

    # L2 normalize
    normed = sk_normalize(cleaned, norm="l2", axis=1)

    tgt_dir.mkdir(parents=True, exist_ok=True)
    np.save(tgt_dir / "word_vectors.npy", normed)
    shutil.copy(src_dir / "vocab.tsv", tgt_dir / "vocab.tsv")

    # Load source meta and update
    with open(src_dir / "meta.json") as f:
        meta = json.load(f)
    meta["version"] = target
    meta["params"]["base_version"] = source
    meta["params"]["postprocess_extra"] = ["remove_pc1", "re_l2"]
    meta["params"]["pc1_singular_values"] = s.tolist()
    with open(tgt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Saved: {tgt_dir}, shape={normed.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    make_nopc1(args.source, args.target)


if __name__ == "__main__":
    main()
