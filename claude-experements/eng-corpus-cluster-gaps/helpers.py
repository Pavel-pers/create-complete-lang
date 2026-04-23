"""
Helper functions for gaps_analysis.ipynb.

Pure functions for loading data, Zipf stats, WordNet coverage, and
benchmark evaluation. More specialized routines live in dedicated modules
(alignment.py, rulsif_utils.py, gap_search.py, wordnet_utils.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.preprocessing import normalize as sk_normalize


# ════════════════════════════════════════════════════════════════
# Loaders (mirror eng-corpus-cluster-explore/helpers.py)
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


def compute_centroids(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-cluster mean vectors, L2-re-normalized. Returns (k, d)."""
    k = int(labels.max() + 1)
    d = vectors.shape[1]
    centroids = np.zeros((k, d), dtype=np.float32)
    for lab in range(k):
        mask = labels == lab
        if mask.any():
            centroids[lab] = vectors[mask].mean(axis=0)
    return sk_normalize(centroids, norm="l2", axis=1)


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
        dists = np.linalg.norm(vectors[indices] - centroids[cid], axis=1)
        top_idx = np.argsort(dists)[:top_k]
        result[cid] = [idx2word[indices[i]] for i in top_idx]
    return result


# ════════════════════════════════════════════════════════════════
# Phase 0: data audit
# ════════════════════════════════════════════════════════════════

def zipf_fit(tf: np.ndarray) -> dict:
    """Fit power-law slope on log-log rank-frequency. Returns dict with slope, intercept, R²."""
    sorted_tf = np.sort(tf)[::-1]
    ranks = np.arange(1, len(sorted_tf) + 1)
    log_r = np.log(ranks)
    log_f = np.log(sorted_tf)
    slope, intercept, r_value, p_value, se = linregress(log_r, log_f)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "n_vocab": int(len(tf)),
    }


def hapax_fraction(tf: np.ndarray) -> float:
    """Fraction of vocabulary with tf==1. For filtered vocab (min_tf>1) this is ~0."""
    return float((tf == 1).sum() / len(tf))


def check_l2_norms(vectors: np.ndarray, tol: float = 1e-4) -> dict:
    """Verify every vector is ~unit-length."""
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "mean": float(norms.mean()),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "std": float(norms.std()),
        "all_close_to_one": bool(np.abs(norms - 1.0).max() < tol),
        "max_abs_deviation": float(np.abs(norms - 1.0).max()),
    }


# ════════════════════════════════════════════════════════════════
# Phase 1: benchmark evaluation
# ════════════════════════════════════════════════════════════════

def spearman_benchmark(
    pairs: list[tuple[str, str, float]],
    word2idx: dict[str, int],
    vectors: np.ndarray,
) -> dict:
    """Spearman correlation between cosine similarity and human scores.

    Vectors assumed L2-normalized so cosine = dot product.
    """
    scores_model = []
    scores_human = []
    missing = 0
    for w1, w2, human in pairs:
        i1 = word2idx.get(w1.lower())
        i2 = word2idx.get(w2.lower())
        if i1 is None or i2 is None:
            missing += 1
            continue
        sim = float(np.dot(vectors[i1], vectors[i2]))
        scores_model.append(sim)
        scores_human.append(human)
    if len(scores_model) < 5:
        return {
            "spearman": float("nan"),
            "p_value": float("nan"),
            "n_pairs": len(scores_model),
            "n_total": len(pairs),
            "coverage": len(scores_model) / max(len(pairs), 1),
            "missing": missing,
        }
    rho, p = spearmanr(scores_model, scores_human)
    return {
        "spearman": float(rho),
        "p_value": float(p),
        "n_pairs": len(scores_model),
        "n_total": len(pairs),
        "coverage": len(scores_model) / len(pairs),
        "missing": missing,
    }


def jaccard_knn(
    vectors_a: np.ndarray,
    vectors_b: np.ndarray,
    k: int = 10,
    n_probe: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Jaccard overlap of k-NN between two embedding spaces over n_probe probe words.

    Vectors must be aligned (same index → same word) and L2-normalized.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(vectors_a)
    probe_idx = rng.choice(n, size=min(n_probe, n), replace=False)
    jaccards = []
    for i in probe_idx:
        sims_a = vectors_a @ vectors_a[i]
        sims_b = vectors_b @ vectors_b[i]
        sims_a[i] = -np.inf
        sims_b[i] = -np.inf
        nn_a = set(np.argsort(-sims_a)[:k].tolist())
        nn_b = set(np.argsort(-sims_b)[:k].tolist())
        inter = len(nn_a & nn_b)
        union = len(nn_a | nn_b)
        jaccards.append(inter / union if union > 0 else 0.0)
    return float(np.mean(jaccards))


# ════════════════════════════════════════════════════════════════
# Phase 2 supporting: DBCV approximation
# ════════════════════════════════════════════════════════════════

def dbcv_approx(
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Approximate DBCV via centroid-distance ratio (Dunn-like).

    Full DBCV (Moulavi et al. 2014) requires MST over all points — infeasible for 79k.
    We use a cheaper proxy: mean over clusters of (inter-centroid-dist − 2·radius)/max(…).
    For our purpose (stop-gate check), this correlates with DBCV sign.
    """
    from sklearn.metrics import silhouette_samples

    if rng is None:
        rng = np.random.default_rng(42)
    if len(vectors) > sample_size:
        idx = rng.choice(len(vectors), sample_size, replace=False)
        v = vectors[idx]
        lab = labels[idx]
    else:
        v = vectors
        lab = labels
    # Use silhouette on sample; centroid-Dunn combined
    try:
        sil = silhouette_samples(v, lab, metric="cosine")
        return float(sil.mean())
    except Exception:
        return float("nan")


def bootstrap_stability(
    vectors: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 20,
    subsample_frac: float = 0.8,
    n_clusters: Optional[int] = None,
    method: str = "kmeans",
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Bootstrap stability: for each cluster in the reference labelling, compute
    max-Jaccard with any cluster in the bootstrap labelling."""
    from sklearn.cluster import BisectingKMeans

    if rng is None:
        rng = np.random.default_rng(42)
    if n_clusters is None:
        n_clusters = int(labels.max() + 1)

    k = n_clusters
    n = len(vectors)
    max_jaccards = [[] for _ in range(k)]
    for it in range(n_iterations):
        sample_idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        v_sub = vectors[sample_idx]
        # Re-cluster
        if method == "kmeans":
            km = BisectingKMeans(n_clusters=k, random_state=42 + it, n_init=1)
            boot_labels = km.fit_predict(v_sub)
        else:
            raise ValueError(method)
        # For each ref cluster, find best match in boot
        ref_labels_sub = labels[sample_idx]
        for c in range(k):
            ref_mask = ref_labels_sub == c
            if ref_mask.sum() == 0:
                continue
            best_j = 0.0
            for b in range(k):
                boot_mask = boot_labels == b
                inter = (ref_mask & boot_mask).sum()
                union = (ref_mask | boot_mask).sum()
                if union > 0:
                    j = inter / union
                    if j > best_j:
                        best_j = j
            max_jaccards[c].append(best_j)
    means = [np.mean(mj) if mj else 0.0 for mj in max_jaccards]
    return {
        "mean_jaccard_per_cluster": means,
        "overall_mean": float(np.mean(means)),
        "stable_frac": float(np.mean([m >= 0.6 for m in means])),
    }
