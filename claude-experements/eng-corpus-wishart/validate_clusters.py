#!/usr/bin/env python3
"""
CLI: Validate clustering quality.

Usage:
    python validate_clusters.py --cluster-version c1 --embeddings-version v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from storage_setup import EXPERIMENT_DATA
from vocab_builder import load_vocab


def uniformity_metrics(labels: np.ndarray, min_required_size: int = 100) -> dict:
    """Compute cluster uniformity metrics."""
    valid = labels[labels >= 0]
    unique, counts = np.unique(valid, return_counts=True)
    n_noise = int((labels == -1).sum())

    if len(counts) == 0:
        return {"n_clusters": 0, "n_noise": n_noise, "pass": False}

    return {
        "n_clusters": len(unique),
        "n_noise": n_noise,
        "n_words_clustered": int(valid.shape[0]),
        "sizes": sorted(counts.tolist()),
        "min_size": int(counts.min()),
        "max_size": int(counts.max()),
        "mean_size": round(float(counts.mean()), 1),
        "std_size": round(float(counts.std()), 1),
        "cv": round(float(counts.std() / counts.mean()), 4) if counts.mean() > 0 else 0,
        "max_min_ratio": round(float(counts.max() / counts.min()), 2) if counts.min() > 0 else float("inf"),
        "n_below_min": int((counts < min_required_size).sum()),
        "pct_below_min": round(float((counts < min_required_size).sum()) / len(counts), 4),
        "pass_all_above_min": bool((counts >= min_required_size).all()),
    }


def linguistic_validation(
    labels: np.ndarray,
    vectors: np.ndarray,
    idx2word: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    For each cluster: compute anchor words, coherence, separation.
    """
    unique_labels = np.unique(labels[labels >= 0])
    clusters = []

    # Compute all centroids first
    centroids = np.zeros((len(unique_labels), vectors.shape[1]))
    label_to_idx = {}
    for ci, lab in enumerate(unique_labels):
        mask = labels == lab
        centroids[ci] = vectors[mask].mean(axis=0)
        label_to_idx[lab] = ci

    for ci, lab in enumerate(unique_labels):
        mask = labels == lab
        cluster_vecs = vectors[mask]
        cluster_words = [idx2word[i] for i in np.where(mask)[0]]
        centroid = centroids[ci]

        # Anchor words: closest to centroid
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        top_indices = np.argsort(dists)[:top_k]
        anchor_words = [cluster_words[i] for i in top_indices]

        # Coherence: average pairwise cosine similarity (sampled for large clusters)
        norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        normed = cluster_vecs / (norms + 1e-10)
        if len(normed) > 200:
            sample_idx = np.random.choice(len(normed), 200, replace=False)
            sample = normed[sample_idx]
        else:
            sample = normed
        sim_matrix = sample @ sample.T
        n = len(sample)
        coherence = float((sim_matrix.sum() - n) / (n * (n - 1))) if n > 1 else 0.0

        # Separation: min distance to other centroids
        other_dists = np.linalg.norm(centroids - centroid, axis=1)
        other_dists[ci] = float("inf")
        separation = float(other_dists.min())

        clusters.append({
            "cluster_id": int(lab),
            "size": int(mask.sum()),
            "anchor_words": anchor_words,
            "coherence": round(coherence, 4),
            "separation": round(separation, 4),
            "sample_words": sorted(cluster_words)[:50],
        })

    clusters.sort(key=lambda x: -x["size"])
    return clusters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate clustering quality.")
    parser.add_argument("--cluster-version", required=True)
    parser.add_argument("--embeddings-version", required=True)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args(argv)

    cluster_dir = EXPERIMENT_DATA / "clusters" / args.cluster_version
    emb_dir = EXPERIMENT_DATA / "embeddings" / args.embeddings_version
    val_dir = EXPERIMENT_DATA / "validation" / args.cluster_version
    val_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    labels_data = np.load(cluster_dir / "labels.npz")
    labels = labels_data["cluster_ids"]
    vectors = np.load(emb_dir / "word_vectors.npy")
    idx2word, _, _, _ = load_vocab(emb_dir / "vocab.tsv")

    print(f"Validating: {args.cluster_version}")
    print(f"Words: {len(labels)}, Vectors: {vectors.shape}")

    # Uniformity
    uni = uniformity_metrics(labels, args.min_cluster_size)
    print(f"\n--- Uniformity ---")
    print(f"Clusters: {uni['n_clusters']}")
    print(f"Sizes: min={uni.get('min_size')}, max={uni.get('max_size')}, "
          f"mean={uni.get('mean_size')}, std={uni.get('std_size')}")
    print(f"CV: {uni.get('cv')}, max/min ratio: {uni.get('max_min_ratio')}")
    print(f"Below minimum ({args.min_cluster_size}): {uni.get('n_below_min')}")
    print(f"Pass: {uni.get('pass_all_above_min')}")

    with open(val_dir / "uniformity_report.json", "w") as f:
        json.dump(uni, f, indent=2)

    # Linguistic
    print(f"\n--- Linguistic Validation ---")
    ling = linguistic_validation(labels, vectors, idx2word, args.top_k)

    for c in ling[:20]:
        print(f"  Cluster {c['cluster_id']} (n={c['size']}): "
              f"anchors={c['anchor_words'][:5]}, "
              f"coherence={c['coherence']:.3f}, sep={c['separation']:.3f}")

    with open(val_dir / "linguistic_report.json", "w") as f:
        json.dump(ling, f, indent=2, ensure_ascii=False)

    # Word lists TSV
    with open(val_dir / "cluster_words.tsv", "w", encoding="utf-8") as f:
        f.write("cluster_id\tsize\tanchor_words\tall_words\n")
        for c in ling:
            anchors = ", ".join(c["anchor_words"][:5])
            all_words = ", ".join(c["sample_words"])
            f.write(f"{c['cluster_id']}\t{c['size']}\t{anchors}\t{all_words}\n")

    print(f"\nReports saved: {val_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
