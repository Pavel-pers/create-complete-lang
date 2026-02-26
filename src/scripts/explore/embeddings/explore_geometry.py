"""Tier 1: zero-resource geometric quality metrics for SVD embeddings.

All metrics are computed solely from the embedding matrix — no labelled data
or external benchmarks required.  The script produces a JSON report with
numeric results and a human-readable quality verdict for each metric.

Metrics
-------
1. Cosine similarity distribution — anisotropy detection.
2. Participation ratio — effective dimensionality.
3. Isotropy — eigenvalue spectrum analysis.
4. Hubness — skewness of k-occurrence distribution.
5. Frequency–norm correlation — whether vector norms encode frequency.
6. kNN symmetry — fraction of reciprocal neighbour pairs.
7. kNN overlap stability — consistency of neighbourhoods across k.

Usage
-----
    python eval_tier1_geometry.py \
        -e artifacts/embeddings.npy \
        -v artifacts/vocab.tsv \
        -o reports/tier1.json

References
----------
- Ethayarajh (2019)          — anisotropy in contextual embeddings.
- Mu & Viswanath (2018)      — isotropy, post-processing.
- Radovanović et al. (2010)  — hubness in high-dimensional spaces.
- Torregrossa et al. (2020)  — participation ratio as quality predictor.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from ._eval_common import (
    EmbeddingData,
    load_embeddings,
    add_common_args,
    write_report
)
from cclang.config.s3 import load_s3_config
from cclang.core.storage import StorageManager, CloudConfig
from cclang.io.fs import LocalConfig


# ======================================================================
# Individual metric functions
# ======================================================================

def cosine_similarity_distribution(
        data: EmbeddingData,
        n_pairs: int = 100_000,
        seed: int = 42,
) -> dict[str, Any]:
    """Sample random pairs and compute cosine similarity statistics.

    A well-formed embedding space should have mean cosine close to zero.
    Values above 0.1 indicate anisotropy: all vectors cluster in a narrow
    cone, reducing discriminative power of cosine similarity.

    The metric follows Ethayarajh (2019).  We use the L2-normalised matrix
    so the dot product equals the cosine.
    """
    rng = np.random.default_rng(seed)
    n = data.vocab_size

    idx1 = rng.integers(0, n, size=n_pairs)
    idx2 = rng.integers(0, n, size=n_pairs)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    sims = np.sum(data.normed[idx1] * data.normed[idx2], axis=1)

    mean_cos = float(np.mean(sims))
    abs_mean = abs(mean_cos)
    if abs_mean < 0.05:
        quality = "GOOD"
    elif abs_mean < 0.15:
        quality = "OK"
    elif abs_mean < 0.30:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "cosine_similarity_distribution",
        "mean": round(mean_cos, 6),
        "std": round(float(np.std(sims)), 6),
        "median": round(float(np.median(sims)), 6),
        "p5": round(float(np.percentile(sims, 5)), 6),
        "p95": round(float(np.percentile(sims, 95)), 6),
        "n_pairs": int(len(idx1)),
        "quality": quality,
        "interpretation": (
            "Mean cosine similarity between random word pairs.  Should be "
            "near 0 for an isotropic space.  |mean| > 0.1 indicates "
            "anisotropy (Ethayarajh 2019)."
        ),
    }


def participation_ratio(data: EmbeddingData) -> dict[str, Any]:
    """Effective dimensionality via the participation ratio.

    PR = (Σ λ_i)² / Σ λ_i²

    where λ_i are eigenvalues of the covariance matrix of centred
    embeddings.  PR ranges from 1 (all variance in one direction) to d
    (uniform).  For k=300, PR > 90 is good; PR < 20 is concerning.

    This metric showed the strongest correlation with downstream task
    performance in Torregrossa et al. (2020).
    """
    cov = np.cov(data.centered.T)  # (k, k)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues[eigenvalues > 0])[::-1]

    pr = float((np.sum(eigenvalues)) ** 2 / np.sum(eigenvalues ** 2))
    d = len(eigenvalues)
    pr_ratio = pr / d

    # Cumulative explained variance
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    if pr_ratio > 0.40:
        quality = "GOOD"
    elif pr_ratio > 0.20:
        quality = "OK"
    elif pr_ratio > 0.10:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "participation_ratio",
        "pr": round(pr, 2),
        "embedding_dim": d,
        "pr_over_dim": round(pr_ratio, 4),
        "dims_for_90pct_var": dims_90,
        "dims_for_95pct_var": dims_95,
        "top1_var_share": round(float(eigenvalues[0] / np.sum(eigenvalues)), 4),
        "top10_var_share": round(float(np.sum(eigenvalues[:10]) / np.sum(eigenvalues)), 4),
        "quality": quality,
        "interpretation": (
            "Effective dimensionality.  PR/dim > 0.3 means the space "
            "uses its dimensions well.  Strongest single predictor of "
            "downstream quality (Torregrossa et al. 2020)."
        ),
    }


def isotropy_metrics(
        data: EmbeddingData,
        n_pairs: int = 100_000,
        seed: int = 42,
) -> dict[str, Any]:
    """Isotropy through eigenvalue ratio and partition-function approximation.

    1) λ_min / λ_max of the covariance spectrum.  Perfectly isotropic = 1.
    2) Mu & Viswanath (2018) partition-function approach: approximate
       Z(c) = (1/N) Σ_w exp(c^T w), then I = min_c Z(c) / max_c Z(c).
       We approximate this with the eigenvectors of the covariance.

    Practical rule: eigenvalue_ratio > 0.01 is acceptable for LSA.
    """
    cov = np.cov(data.centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    eigen_ratio = float(eigenvalues[-1] / eigenvalues[0])

    # Effective rank (Roy & Vetterli 2007):
    # erank = exp(H(p))  where p_i = σ_i / Σσ_i  and H is Shannon entropy
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    effective_rank = float(np.exp(entropy))

    # Dominance of top component
    top1_dominance = float(eigenvalues[0] / np.sum(eigenvalues))

    if eigen_ratio > 0.01:
        quality = "GOOD"
    elif eigen_ratio > 0.001:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "isotropy",
        "eigenvalue_ratio_min_max": round(eigen_ratio, 8),
        "effective_rank": round(effective_rank, 2),
        "top1_dominance": round(top1_dominance, 4),
        "condition_number": round(float(eigenvalues[0] / (eigenvalues[-1] + 1e-30)), 2),
        "quality": quality,
        "interpretation": (
            "Eigenvalue ratio λ_min/λ_max measures how uniformly the space "
            "uses all directions.  Effective rank (Roy & Vetterli 2007) gives "
            "the exponential Shannon entropy of normalised eigenvalues.  "
            "Higher is better."
        ),
    }


def hubness_analysis(
        data: EmbeddingData,
        k: int = 10,
        sample_size: int | None = None,
        seed: int = 42,
) -> dict[str, Any]:
    """Hubness: skewness of k-occurrence distribution.

    In high-dimensional spaces, a few "hub" points appear as nearest
    neighbours of disproportionately many other points, distorting
    similarity search.  The skewness S_N of the k-occurrence count
    N_k(x) quantifies this.

    S_N < 1.0 is healthy; S_N > 2.5 indicates serious problems.
    (Radovanović et al. 2010)
    """
    X = data.normed
    n = len(X)

    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]
        n = len(X)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # exclude self

    # k-occurrence: how many times each point is a neighbour
    n_k = np.zeros(n, dtype=np.int64)
    for row in neighbor_indices:
        for j in row:
            n_k[j] += 1

    skewness = float(stats.skew(n_k))
    mean_nk = float(np.mean(n_k))
    std_nk = float(np.std(n_k))

    hub_threshold = mean_nk + 2 * std_nk
    n_hubs = int(np.sum(n_k > hub_threshold))
    n_antihubs = int(np.sum(n_k == 0))

    if skewness < 1.0:
        quality = "GOOD"
    elif skewness < 1.5:
        quality = "OK"
    elif skewness < 2.5:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "hubness",
        "skewness_Sn": round(skewness, 4),
        "k": k,
        "sample_size": n,
        "mean_k_occurrence": round(mean_nk, 2),
        "std_k_occurrence": round(std_nk, 2),
        "n_hubs": n_hubs,
        "n_antihubs": n_antihubs,
        "hub_fraction": round(n_hubs / n, 4),
        "antihub_fraction": round(n_antihubs / n, 4),
        "quality": quality,
        "interpretation": (
            "Skewness of the k-occurrence distribution.  S_N < 1.0 healthy; "
            "> 2.5 means a few 'hub' words appear as neighbours of everything "
            "(Radovanović et al. 2010)."
        ),
    }


def frequency_norm_correlation(data: EmbeddingData) -> dict[str, Any]:
    """Spearman ρ between log-frequency and L2-norm of raw embeddings.

    If |ρ| > 0.6, vector norms are dominated by word frequency rather than
    semantics.  Consider L2-normalising all vectors after SVD.
    (Mu & Viswanath 2018)
    """
    if data.tf is None:
        return {
            "metric": "frequency_norm_correlation",
            "error": "term frequencies not available in vocab",
        }

    norms = np.linalg.norm(data.raw, axis=1)
    log_tf = np.log1p(data.tf.astype(np.float64))

    # Mask zero-frequency terms (should not exist, but just in case)
    valid = data.tf > 0
    if valid.sum() < 10:
        return {
            "metric": "frequency_norm_correlation",
            "error": "too few terms with nonzero frequency",
        }

    sp_r, sp_p = stats.spearmanr(log_tf[valid], norms[valid])
    pe_r, pe_p = stats.pearsonr(log_tf[valid], norms[valid])

    abs_r = abs(float(sp_r))
    if abs_r < 0.20:
        quality = "GOOD"
    elif abs_r < 0.40:
        quality = "OK"
    elif abs_r < 0.60:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "frequency_norm_correlation",
        "spearman_rho": round(float(sp_r), 4),
        "spearman_p": float(sp_p),
        "pearson_r": round(float(pe_r), 4),
        "pearson_p": float(pe_p),
        "quality": quality,
        "interpretation": (
            "Spearman ρ between log(tf) and ‖v‖₂.  |ρ| < 0.3 is good; "
            "|ρ| > 0.6 means norms are driven by frequency, not semantics "
            "(Mu & Viswanath 2018).  Fix: L2-normalise all vectors."
        ),
    }


def knn_symmetry(
        data: EmbeddingData,
        k: int = 10,
        sample_size: int = 5_000,
        seed: int = 42,
) -> dict[str, Any]:
    """Fraction of reciprocal nearest-neighbour pairs.

    If A is in kNN(B), is B also in kNN(A)?  High symmetry (> 0.5) means
    consistent, well-structured local neighbourhoods.
    """
    X = data.normed
    n = len(X)

    if n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]
        n = len(X)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbors = indices[:, 1:]

    neighbor_sets = [set(row) for row in neighbors]

    symmetric = 0
    total = 0
    for i, neigh_set in enumerate(neighbor_sets):
        for j in neigh_set:
            total += 1
            if i in neighbor_sets[j]:
                symmetric += 1

    ratio = symmetric / total if total > 0 else 0.0

    if ratio > 0.50:
        quality = "GOOD"
    elif ratio > 0.35:
        quality = "OK"
    elif ratio > 0.20:
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "knn_symmetry",
        "symmetry_ratio": round(ratio, 4),
        "k": k,
        "sample_size": n,
        "quality": quality,
        "interpretation": (
            "Fraction of (A,B) pairs where A∈kNN(B) AND B∈kNN(A).  "
            "> 0.5 indicates coherent local structure."
        ),
    }


def knn_overlap_stability(
        data: EmbeddingData,
        k_values: tuple[int, ...] = (5, 10, 20),
        sample_size: int = 3_000,
        seed: int = 42,
) -> dict[str, Any]:
    """Check that top-k neighbours are stable as k grows.

    For each word, we check: what fraction of its top-k_small neighbours
    are also present in its top-k_large neighbours?  High containment
    (> 0.8) means the similarity ranking is robust.
    """
    X = data.normed
    n = len(X)

    if n > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]
        n = len(X)

    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    containment_scores: dict[str, float] = {}
    for i in range(len(k_values) - 1):
        k_small = k_values[i]
        k_large = k_values[i + 1]

        nn_small = [set(row[1:k_small + 1]) for row in indices]
        nn_large = [set(row[1:k_large + 1]) for row in indices]

        scores = [
            len(ns & nl) / len(ns) if len(ns) > 0 else 1.0
            for ns, nl in zip(nn_small, nn_large)
        ]
        key = f"k{k_small}_in_k{k_large}"
        containment_scores[key] = round(float(np.mean(scores)), 4)

    values = list(containment_scores.values())
    if all(v > 0.85 for v in values):
        quality = "GOOD"
    elif all(v > 0.65 for v in values):
        quality = "OK"
    elif all(v > 0.50 for v in values):
        quality = "MODERATE"
    else:
        quality = "POOR"

    return {
        "metric": "knn_overlap_stability",
        "containment": containment_scores,
        "k_values": list(k_values),
        "sample_size": n,
        "quality": quality,
        "interpretation": (
            "Fraction of top-k_small neighbours that are also in top-k_large.  "
            "> 0.8 for all pairs means stable similarity rankings."
        ),
    }


# ======================================================================
# Orchestrator
# ======================================================================

def run_tier1(
        data: EmbeddingData,
        *,
        n_pairs: int = 100_000,
        hubness_k: int = 10,
        hubness_sample: int | None = None,
        knn_k: int = 10,
        knn_sample: int = 5_000,
        overlap_k_values: tuple[int, ...] = (5, 10, 20),
        overlap_sample: int = 3_000,
        seed: int = 42,
) -> dict[str, Any]:
    """Run all Tier 1 metrics and return a combined report."""
    t0 = time.time()
    metrics: list[dict[str, Any]] = []

    print("  [1/7] cosine similarity distribution …", file=sys.stderr)
    metrics.append(cosine_similarity_distribution(data, n_pairs=n_pairs, seed=seed))

    print("  [2/7] participation ratio …", file=sys.stderr)
    metrics.append(participation_ratio(data))

    print("  [3/7] isotropy …", file=sys.stderr)
    metrics.append(isotropy_metrics(data))

    print("  [4/7] hubness …", file=sys.stderr)
    metrics.append(hubness_analysis(
        data, k=hubness_k, sample_size=hubness_sample, seed=seed,
    ))

    print("  [5/7] frequency–norm correlation …", file=sys.stderr)
    metrics.append(frequency_norm_correlation(data))

    print("  [6/7] kNN symmetry …", file=sys.stderr)
    metrics.append(knn_symmetry(data, k=knn_k, sample_size=knn_sample, seed=seed))

    print("  [7/7] kNN overlap stability …", file=sys.stderr)
    metrics.append(knn_overlap_stability(
        data, k_values=overlap_k_values, sample_size=overlap_sample, seed=seed,
    ))

    elapsed = time.time() - t0

    # Summary
    quality_map = {m["metric"]: m.get("quality", "N/A") for m in metrics}
    counts = {}
    for q in quality_map.values():
        counts[q] = counts.get(q, 0) + 1

    return {
        "tier": 1,
        "name": "zero_resource_geometry",
        "embedding_shape": list(data.raw.shape),
        "elapsed_sec": round(elapsed, 2),
        "metrics": metrics,
        "summary": {
            "per_metric": quality_map,
            "counts": counts,
        },
    }


# ======================================================================
# CLI
# ======================================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tier 1: zero-resource geometric quality metrics for SVD embeddings."
    )

    add_common_args(parser)

    parser.add_argument("--n-pairs", type=int, default=100_000,
                        help="Random pairs for cosine distribution (default: 100000)")
    parser.add_argument("--hubness-k", type=int, default=10,
                        help="k for hubness analysis (default: 10)")
    parser.add_argument("--hubness-sample", type=int, default=None,
                        help="Sample size for hubness (default: all)")
    parser.add_argument("--knn-k", type=int, default=10,
                        help="k for kNN symmetry (default: 10)")
    parser.add_argument("--knn-sample", type=int, default=5_000,
                        help="Sample size for kNN symmetry (default: 5000)")
    parser.add_argument("--overlap-sample", type=int, default=3_000,
                        help="Sample size for overlap stability (default: 3000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args(argv)

    base_path = args.base_path
    local_cfg = LocalConfig(
        base_path=base_path,
        save_local=True,
        cache_files=True,
        temp_base=base_path / "temp",
    )
    s3_cfg = load_s3_config()
    cloud_cfg = CloudConfig(
        enable=s3_cfg.enable,
        base_path=Path('data'),
        s3_config=s3_cfg,
        max_upload_threads=0,
    )
    storage = StorageManager(local_cfg, cloud_cfg)

    print(f"Loading embeddings from {args.embeddings_path} …", file=sys.stderr)
    data = load_embeddings(storage, args.vocab_path, args.embeddings_path)
    print(
        f"  shape: {data.raw.shape}  vocab: {data.vocab_size}",
        file=sys.stderr,
    )

    print("Running Tier 1 evaluation …", file=sys.stderr)
    report = run_tier1(
        data,
        n_pairs=args.n_pairs,
        hubness_k=args.hubness_k,
        hubness_sample=args.hubness_sample,
        knn_k=args.knn_k,
        knn_sample=args.knn_sample,
        overlap_sample=args.overlap_sample,
        seed=args.seed,
    )


    write_report(report, args.output)
    if args.output:
        print(f"Report written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
