"""
Helper functions for cluster exploration notebook.

All functions are pure — take numpy arrays explicitly, return values.
Used by cluster_explore.ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)
from sklearn.preprocessing import normalize as sk_normalize


# ════════════════════════════════════════════════════════════════
# Loaders
# ════════════════════════════════════════════════════════════════

def load_embeddings(emb_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Load word vectors + vocab. Returns (vectors, idx2word, tf, df)."""
    vectors = np.load(emb_dir / "word_vectors.npy").astype(np.float32)
    idx2word: list[str] = []
    tf_list, df_list = [], []
    with open(emb_dir / "vocab.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx2word.append(parts[1])
            tf_list.append(int(parts[2]))
            df_list.append(int(parts[3]))
    return vectors, idx2word, np.array(tf_list), np.array(df_list)


def load_level(clusters_dir: Path, version: str) -> dict:
    """Load one clustering level (labels + meta)."""
    vdir = clusters_dir / version
    data = np.load(vdir / "labels.npz")
    labels = data["cluster_ids"]
    densities = data["densities"]
    roles = data["roles"]
    with open(vdir / "meta.json") as f:
        meta = json.load(f)
    return {
        "version": version,
        "labels": labels,
        "densities": densities,
        "roles": roles,
        "n_clusters": int(labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0,
        "meta": meta,
    }


# ════════════════════════════════════════════════════════════════
# Per-cluster arithmetic
# ════════════════════════════════════════════════════════════════

def compute_centroids(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-cluster mean vectors, L2-re-normalized. Returns (k, d)."""
    k = int(labels.max() + 1)
    d = vectors.shape[1]
    centroids = np.zeros((k, d), dtype=np.float32)
    for lab in range(k):
        mask = labels == lab
        if mask.any():
            centroids[lab] = vectors[mask].mean(axis=0)
    centroids = sk_normalize(centroids, norm="l2", axis=1)
    return centroids


def compute_per_cluster_stats(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    densities: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    For each cluster compute: size, inertia, mean/median/max distance to centroid,
    mean density.
    """
    rows = []
    k = centroids.shape[0]
    for cid in range(k):
        mask = labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        vecs = vectors[mask]
        dists = np.linalg.norm(vecs - centroids[cid], axis=1)
        row = {
            "cluster_id": cid,
            "size": n,
            "inertia": float((dists ** 2).sum()),
            "mean_dist": float(dists.mean()),
            "median_dist": float(np.median(dists)),
            "max_dist": float(dists.max()),
            "std_dist": float(dists.std()),
        }
        if densities is not None:
            row["mean_density"] = float(densities[mask].mean())
            row["median_density"] = float(np.median(densities[mask]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster_id")


def centroid_distance_matrix(centroids: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Pairwise distance matrix between centroids."""
    if metric == "cosine":
        # Centroids are L2-normalized, so cosine distance = 1 - dot product
        sim = centroids @ centroids.T
        np.clip(sim, -1.0, 1.0, out=sim)
        return 1.0 - sim
    else:
        from scipy.spatial.distance import squareform, pdist
        return squareform(pdist(centroids, metric=metric))


def anchor_words(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    idx2word: list[str],
    top_k: int = 5,
) -> dict[int, list[str]]:
    """For each cluster, find top_k words closest to centroid."""
    result: dict[int, list[str]] = {}
    k = centroids.shape[0]
    for cid in range(k):
        mask = labels == cid
        if not mask.any():
            result[cid] = []
            continue
        indices = np.where(mask)[0]
        vecs = vectors[indices]
        dists = np.linalg.norm(vecs - centroids[cid], axis=1)
        top_idx = np.argsort(dists)[:top_k]
        result[cid] = [idx2word[indices[i]] for i in top_idx]
    return result


# ════════════════════════════════════════════════════════════════
# Quality metrics
# ════════════════════════════════════════════════════════════════

def stratified_sample_indices(
    labels: np.ndarray,
    total_size: int,
    min_per_cluster: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Stratified sampling: ensure each cluster has at least min_per_cluster samples.
    Returns array of indices into the original labels array.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    unique = np.unique(labels[labels >= 0])
    k = len(unique)
    n = len(labels)

    # Target per cluster proportional to size, but with floor
    sizes = np.array([(labels == cid).sum() for cid in unique])
    target = np.maximum(
        min_per_cluster,
        np.round(total_size * sizes / n).astype(int),
    )
    target = np.minimum(target, sizes)  # can't take more than available

    indices_list = []
    for i, cid in enumerate(unique):
        cluster_indices = np.where(labels == cid)[0]
        n_take = target[i]
        if n_take >= len(cluster_indices):
            indices_list.append(cluster_indices)
        else:
            picked = rng.choice(cluster_indices, size=n_take, replace=False)
            indices_list.append(picked)

    return np.concatenate(indices_list)


def silhouette_sampled(
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10_000,
    min_per_cluster: int = 20,
    metric: str = "cosine",
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute silhouette on a stratified sample.
    Returns (mean_score, per_sample_scores, sample_indices).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sample_idx = stratified_sample_indices(labels, sample_size, min_per_cluster, rng)
    sample_vecs = vectors[sample_idx]
    sample_labels = labels[sample_idx]

    scores = silhouette_samples(sample_vecs, sample_labels, metric=metric)
    return float(scores.mean()), scores, sample_idx


def silhouette_per_cluster(
    per_sample_scores: np.ndarray,
    sample_labels: np.ndarray,
) -> dict[int, float]:
    """Mean silhouette for each cluster from per-sample scores."""
    result = {}
    for cid in np.unique(sample_labels):
        mask = sample_labels == cid
        result[int(cid)] = float(per_sample_scores[mask].mean())
    return result


def davies_bouldin_full(vectors: np.ndarray, labels: np.ndarray) -> float:
    return float(davies_bouldin_score(vectors, labels))


def calinski_harabasz_full(vectors: np.ndarray, labels: np.ndarray) -> float:
    return float(calinski_harabasz_score(vectors, labels))


def dunn_index_centroid(
    centroids: np.ndarray,
    max_radii: np.ndarray,
    metric: str = "cosine",
) -> float:
    """
    Centroid-based Dunn approximation:
    Dunn = min(inter-centroid dist) / max(2 * radius_i)

    Exact Dunn uses full pairwise intra/inter — too expensive for large k.
    """
    dist_matrix = centroid_distance_matrix(centroids, metric=metric)
    k = centroids.shape[0]
    mask = ~np.eye(k, dtype=bool)
    inter_min = dist_matrix[mask].min()
    intra_max = (2 * max_radii).max()
    if intra_max == 0:
        return float("inf")
    return float(inter_min / intra_max)


# ════════════════════════════════════════════════════════════════
# Distributional metrics
# ════════════════════════════════════════════════════════════════

def gini(sizes: np.ndarray) -> float:
    """Gini coefficient of cluster sizes. 0 = equal, 1 = fully concentrated."""
    sizes = np.sort(sizes)
    n = len(sizes)
    if n == 0 or sizes.sum() == 0:
        return 0.0
    cum = np.cumsum(sizes, dtype=float)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def entropy_sizes(sizes: np.ndarray) -> float:
    """Shannon entropy of size distribution (in nats)."""
    p = sizes / sizes.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def entropy_sizes_normalized(sizes: np.ndarray) -> float:
    """Normalized Shannon entropy: entropy / log(k). 1 = perfectly uniform."""
    e = entropy_sizes(sizes)
    k = len(sizes)
    return e / np.log(k) if k > 1 else 0.0


def outlier_fraction(sizes: np.ndarray, k_sigma: float = 2.0) -> float:
    """Fraction of clusters with size > mean + k_sigma*std or < mean - k_sigma*std."""
    mean, std = sizes.mean(), sizes.std()
    mask = (sizes > mean + k_sigma * std) | (sizes < mean - k_sigma * std)
    return float(mask.mean())


def distribution_summary(sizes: np.ndarray) -> dict:
    """All distributional metrics in one dict."""
    return {
        "n_clusters": len(sizes),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "std": float(sizes.std()),
        "cv": float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else 0.0,
        "max_min_ratio": float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf"),
        "gini": gini(sizes),
        "entropy": entropy_sizes(sizes),
        "entropy_norm": entropy_sizes_normalized(sizes),
        "outlier_frac_2sigma": outlier_fraction(sizes, 2.0),
    }


# ════════════════════════════════════════════════════════════════
# Selection helpers
# ════════════════════════════════════════════════════════════════

def pick_top_and_random(
    sizes: np.ndarray,
    top_n: int = 20,
    random_n: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Select top_n largest + random_n random (non-overlapping).
    Returns dict with top, random, combined (top-first), group labels.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    order = np.argsort(-sizes)  # descending by size
    top = order[:top_n].tolist()

    remaining = [i for i in range(len(sizes)) if i not in set(top)]
    if len(remaining) <= random_n:
        random_pick = remaining
    else:
        random_pick = rng.choice(remaining, size=random_n, replace=False).tolist()

    combined = top + random_pick
    group = np.array(["top"] * len(top) + ["random"] * len(random_pick))

    return {
        "top": top,
        "random": random_pick,
        "combined": combined,
        "group": group,
    }


# ════════════════════════════════════════════════════════════════
# UMAP projection
# ════════════════════════════════════════════════════════════════

def run_umap(
    vectors: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> tuple[np.ndarray, "umap.UMAP"]:
    """
    Fit UMAP on vectors. Returns (2D_coords, fitted_reducer).
    If cache_path is provided and exists, load from cache (only coords; reducer not cached).
    """
    import umap

    if cache_path is not None and cache_path.exists() and not force:
        coords = np.load(cache_path)
        print(f"[umap] loaded cached coords from {cache_path}")
        # Still need to fit reducer for centroid projection
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            metric=metric, random_state=random_state,
        )
        reducer.fit(vectors)
        return coords, reducer

    print(f"[umap] fitting UMAP (n={len(vectors)}, d={vectors.shape[1]}, nn={n_neighbors}) ...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=random_state,
    )
    coords = reducer.fit_transform(vectors)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, coords)
        print(f"[umap] cached to {cache_path}")

    return coords, reducer


def project_centroids(reducer, centroids: np.ndarray) -> np.ndarray:
    """Project centroids into the fitted UMAP 2D space."""
    return reducer.transform(centroids)


# ════════════════════════════════════════════════════════════════
# Cross-level flow
# ════════════════════════════════════════════════════════════════

def build_flow_matrix(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    k_a: int,
    k_b: int,
) -> np.ndarray:
    """Count of points in (A_i, B_j) intersection. Returns (k_a, k_b) int32."""
    flow = np.zeros((k_a, k_b), dtype=np.int32)
    np.add.at(flow, (labels_a, labels_b), 1)
    return flow


def purity_from_flow(flow: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Purity of each column (axis=0) or row (axis=1):
    max count in that slice / total in that slice.
    """
    totals = flow.sum(axis=axis)
    maxes = flow.max(axis=axis)
    purity = np.zeros_like(totals, dtype=np.float32)
    mask = totals > 0
    purity[mask] = maxes[mask] / totals[mask]
    return purity
