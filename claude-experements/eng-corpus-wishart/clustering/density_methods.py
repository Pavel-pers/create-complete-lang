"""
Density-based clustering methods: DBSCAN and HDBSCAN.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import DBSCAN


def build_dbscan(
    vectors: np.ndarray,
    eps: float = 0.2,
    min_samples: int = 10,
    metric: str = "euclidean",
    assign_noise: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """DBSCAN clustering."""
    t0 = time.monotonic()
    n = vectors.shape[0]
    print(f"[dbscan] n={n}, eps={eps}, min_samples={min_samples} ...")

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = model.fit_predict(vectors)

    n_noise = int((labels == -1).sum())
    unique = np.unique(labels[labels >= 0])
    n_clusters = len(unique)

    if assign_noise and n_noise > 0 and n_clusters > 0:
        # Assign noise to nearest cluster centroid
        centroids = np.zeros((n_clusters, vectors.shape[1]))
        label_map = {lab: idx for idx, lab in enumerate(unique)}
        for lab in unique:
            centroids[label_map[lab]] = vectors[labels == lab].mean(axis=0)
        noise_idx = np.where(labels == -1)[0]
        # Batched distance
        for start in range(0, len(noise_idx), 5000):
            batch = noise_idx[start:start + 5000]
            dists = np.linalg.norm(
                vectors[batch][:, None, :] - centroids[None, :, :], axis=2
            )
            nearest = np.argmin(dists, axis=1)
            for i, ni in enumerate(batch):
                labels[ni] = unique[nearest[i]]

    # Relabel sequentially
    unique_final = np.unique(labels)
    unique_final = unique_final[unique_final >= 0]
    mapping = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.array([mapping.get(l, -1) for l in labels], dtype=np.int32)

    counts = np.bincount(labels_out[labels_out >= 0])
    elapsed = time.monotonic() - t0

    meta = {
        "method": "dbscan",
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "n_clusters": int(len(counts)),
        "n_noise_before_assign": n_noise,
        "cluster_sizes": {
            "min": int(counts.min()) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
            "mean": round(float(counts.mean()), 1) if len(counts) else 0,
            "std": round(float(counts.std()), 1) if len(counts) else 0,
        },
        "total_time_sec": round(elapsed, 1),
    }

    print(f"[dbscan] {len(counts)} clusters, sizes: "
          f"[{counts.min() if len(counts) else 0}-{counts.max() if len(counts) else 0}], "
          f"noise={n_noise}, time={elapsed:.1f}s")

    densities = np.zeros(n, dtype=np.float32)
    return labels_out, densities, meta


def build_hdbscan(
    vectors: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    metric: str = "euclidean",
    assign_noise: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """HDBSCAN clustering."""
    from hdbscan import HDBSCAN

    t0 = time.monotonic()
    n = vectors.shape[0]
    print(f"[hdbscan] n={n}, min_cluster_size={min_cluster_size}, "
          f"method={cluster_selection_method} ...")

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        core_dist_n_jobs=-1,
    )
    labels = model.fit_predict(vectors.astype(np.float64))

    n_noise = int((labels == -1).sum())
    unique = np.unique(labels[labels >= 0])
    n_clusters = len(unique)

    if assign_noise and n_noise > 0 and n_clusters > 0:
        centroids = np.zeros((n_clusters, vectors.shape[1]))
        label_map = {lab: idx for idx, lab in enumerate(unique)}
        for lab in unique:
            centroids[label_map[lab]] = vectors[labels == lab].mean(axis=0)
        noise_idx = np.where(labels == -1)[0]
        for start in range(0, len(noise_idx), 5000):
            batch = noise_idx[start:start + 5000]
            dists = np.linalg.norm(
                vectors[batch][:, None, :] - centroids[None, :, :], axis=2
            )
            nearest = np.argmin(dists, axis=1)
            for i, ni in enumerate(batch):
                labels[ni] = unique[nearest[i]]

    unique_final = np.unique(labels)
    unique_final = unique_final[unique_final >= 0]
    mapping = {old: new for new, old in enumerate(unique_final)}
    labels_out = np.array([mapping.get(l, -1) for l in labels], dtype=np.int32)

    counts = np.bincount(labels_out[labels_out >= 0])
    elapsed = time.monotonic() - t0

    # Use probabilities_ as density proxy if available
    try:
        densities = model.probabilities_.astype(np.float32)
    except AttributeError:
        densities = np.zeros(n, dtype=np.float32)

    meta = {
        "method": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": cluster_selection_method,
        "metric": metric,
        "n_clusters": int(len(counts)),
        "n_noise_before_assign": n_noise,
        "cluster_sizes": {
            "min": int(counts.min()) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
            "mean": round(float(counts.mean()), 1) if len(counts) else 0,
            "std": round(float(counts.std()), 1) if len(counts) else 0,
        },
        "total_time_sec": round(elapsed, 1),
    }

    print(f"[hdbscan] {len(counts)} clusters, sizes: "
          f"[{counts.min() if len(counts) else 0}-{counts.max() if len(counts) else 0}], "
          f"noise={n_noise}, time={elapsed:.1f}s")

    return labels_out, densities, meta
