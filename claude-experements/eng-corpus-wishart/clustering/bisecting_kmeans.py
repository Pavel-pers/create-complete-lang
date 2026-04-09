"""
Bisecting K-Means — recursively split the largest cluster until reaching target count.

This naturally produces more uniform clusters because at each step we split
the largest cluster, redistributing its mass.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans


def build(
    vectors: np.ndarray,
    n_clusters: int = 18,
    max_iter: int = 300,
    random_state: int = 42,
    max_size_ratio: float = 5.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Bisecting K-Means: start with 1 cluster, recursively split the largest.

    Parameters
    ----------
    max_size_ratio : continue splitting even after reaching n_clusters
                     if max/min ratio exceeds this value.
    """
    t0 = time.monotonic()
    n = vectors.shape[0]

    print(f"[bisect-kmeans] n={n}, target_k={n_clusters} ...")

    labels = np.zeros(n, dtype=np.int32)
    next_label = 1

    def split_cluster(cluster_label: int) -> int:
        nonlocal next_label
        mask = labels == cluster_label
        cluster_vecs = vectors[mask]
        cluster_indices = np.where(mask)[0]

        km = KMeans(n_clusters=2, init="k-means++", n_init=5,
                    max_iter=max_iter, random_state=random_state)
        sub_labels = km.fit_predict(cluster_vecs)

        # Assign new label to the second half
        new_label = next_label
        next_label += 1
        for i, idx in enumerate(cluster_indices):
            if sub_labels[i] == 1:
                labels[idx] = new_label

        return new_label

    min_split_size = kwargs.get("min_cluster_size", 100) * 2

    # Phase 1: split until we reach n_clusters
    while len(np.unique(labels)) < n_clusters:
        unique, counts = np.unique(labels, return_counts=True)
        # Only split clusters large enough
        splittable = counts >= min_split_size
        if not splittable.any():
            print(f"[bisect-kmeans] no more splittable clusters (min_split={min_split_size})")
            break
        # Pick largest splittable cluster
        best_idx = np.argmax(counts * splittable)
        largest_label = unique[best_idx]
        split_cluster(largest_label)

        k_now = len(np.unique(labels))
        if k_now % 10 == 0:
            print(f"[bisect-kmeans] {k_now}/{n_clusters} clusters")

    # Phase 2: if max/min ratio too high, split oversized clusters
    # but only up to 2x the target count
    max_extra = n_clusters
    extra_splits = 0
    for _ in range(max_extra):
        unique, counts = np.unique(labels, return_counts=True)
        ratio = counts.max() / (counts.mean() + 1e-10)
        if ratio <= max_size_ratio or counts.max() < 300:
            break
        largest_label = unique[np.argmax(counts)]
        split_cluster(largest_label)
        extra_splits += 1

    if extra_splits:
        print(f"[bisect-kmeans] +{extra_splits} extra splits for balance")

    # Phase 3: merge small clusters into nearest large neighbor
    min_size = kwargs.get("min_cluster_size", 100)
    while True:
        unique_labs, cnts = np.unique(labels, return_counts=True)
        small_mask = cnts < min_size
        if not small_mask.any():
            break
        large_labels = unique_labs[~small_mask]
        if len(large_labels) == 0:
            break
        large_centroids = np.zeros((len(large_labels), vectors.shape[1]))
        for idx, lab in enumerate(large_labels):
            large_centroids[idx] = vectors[labels == lab].mean(axis=0)
        merged_any = False
        for lab in unique_labs[small_mask]:
            sc = vectors[labels == lab].mean(axis=0)
            dists = np.linalg.norm(large_centroids - sc, axis=1)
            nearest = large_labels[np.argmin(dists)]
            labels[labels == lab] = nearest
            merged_any = True
        if not merged_any:
            break

    # Relabel sequentially
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    labels = np.array([mapping[l] for l in labels], dtype=np.int32)

    # Compute densities
    unique, counts = np.unique(labels, return_counts=True)
    centroids = np.zeros((len(unique), vectors.shape[1]))
    for i, lab in enumerate(unique):
        centroids[i] = vectors[labels == lab].mean(axis=0)
    densities = np.linalg.norm(vectors - centroids[labels], axis=1).astype(np.float32)

    elapsed = time.monotonic() - t0

    meta = {
        "method": "bisecting-kmeans",
        "n_clusters": int(len(unique)),
        "target_clusters": n_clusters,
        "max_size_ratio": max_size_ratio,
        "cluster_sizes": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": round(float(counts.mean()), 1),
            "std": round(float(counts.std()), 1),
        },
        "total_time_sec": round(elapsed, 1),
    }

    print(f"[bisect-kmeans] {len(unique)} clusters, sizes: [{counts.min()}-{counts.max()}], "
          f"ratio={counts.max()/counts.min():.1f}, time={elapsed:.1f}s")

    return labels, densities, meta
