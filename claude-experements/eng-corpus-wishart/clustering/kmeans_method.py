"""
Spherical K-Means clustering.

For L2-normalized vectors, standard K-Means is equivalent to Spherical K-Means
(minimizing cosine distance). This produces much more uniform cluster sizes
compared to density-based methods like Wishart.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def build(
    vectors: np.ndarray,
    n_clusters: int = 18,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    use_minibatch: bool = False,
    batch_size: int = 4096,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Spherical K-Means clustering.

    Returns (labels, densities, meta_dict).
    densities = distance to cluster centroid for each point.
    """
    t0 = time.monotonic()
    n = vectors.shape[0]

    print(f"[kmeans] clustering n={n}, k={n_clusters}, init={init} ...")

    if use_minibatch:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            batch_size=batch_size,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )

    labels = model.fit_predict(vectors)
    centroids = model.cluster_centers_

    # Compute distance to centroid for each point (as density proxy)
    densities = np.linalg.norm(vectors - centroids[labels], axis=1).astype(np.float32)

    unique, counts = np.unique(labels, return_counts=True)

    elapsed = time.monotonic() - t0

    meta = {
        "method": "kmeans",
        "n_clusters": int(len(unique)),
        "init": init,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": random_state,
        "use_minibatch": use_minibatch,
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "cluster_sizes": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": round(float(counts.mean()), 1),
            "std": round(float(counts.std()), 1),
        },
        "total_time_sec": round(elapsed, 1),
    }

    print(f"[kmeans] {len(unique)} clusters, sizes: [{counts.min()}-{counts.max()}], "
          f"inertia={model.inertia_:.1f}, time={elapsed:.1f}s")

    return labels.astype(np.int32), densities, meta
