"""
RuLSIF wrapper and synthetic benchmark.

Uses `densratio` library for α-relative density ratio estimation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def run_rulsif(
    X_p: np.ndarray,
    X_q: np.ndarray,
    alpha: float = 0.1,
    kernel_num: int = 200,
    verbose: bool = False,
) -> object:
    """Fit α-relative density ratio model via densratio.

    Returns fitted model with .compute_density_ratio(x) method.
    p = source distribution (numerator)
    q = reference distribution (denominator)
    r_α(x) = p(x) / (α·p(x) + (1-α)·q(x))
    """
    from densratio import densratio

    # densratio expects (n_nu, d) and (n_de, d): x in nu (numerator=p), y in de (denom=q)
    # Its API: densratio(x, y, alpha=α) → model
    # alpha here matches our α convention (small → closer to p/q)
    result = densratio(
        x=np.asarray(X_p, dtype=np.float64),
        y=np.asarray(X_q, dtype=np.float64),
        alpha=alpha,
        kernel_num=kernel_num,
        verbose=verbose,
    )
    return result


def synthetic_gap_benchmark(
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    cid: int,
    hole_fraction: float = 0.3,
    hole_radius: float = 0.3,
    alpha: float = 0.1,
    d_reduce: int = 30,
    kernel_num: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Create a synthetic gap in cluster `cid` and measure RuLSIF detection AUC.

    Procedure:
    1. Extract cluster's points (P = full).
    2. Pick a random anchor point, remove all points within `hole_radius` cosine distance,
       up to `hole_fraction` of total.
    3. Q = cluster without those points.
    4. Run RuLSIF(P, Q) → r_α(x) for each point x in P.
    5. AUC = area under ROC of r_α distinguishing removed from kept.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng(42)

    mask = cluster_labels == cid
    cluster_vecs = vectors[mask]
    n = len(cluster_vecs)
    if n < 80:
        return {"auc": float("nan"), "n_removed": 0, "n_cluster": n, "error": "cluster too small"}

    # Pick anchor index
    anchor_idx = int(rng.choice(n))
    anchor = cluster_vecs[anchor_idx]

    # Compute cosine distances from anchor (vectors assumed L2-normalized)
    dists = 1.0 - cluster_vecs @ anchor

    # Build hole: points with dist < hole_radius, capped at hole_fraction
    candidate_mask = dists < hole_radius
    n_hole_max = int(n * hole_fraction)
    cand_idx = np.where(candidate_mask)[0]
    if len(cand_idx) > n_hole_max:
        cand_idx = cand_idx[np.argsort(dists[cand_idx])[:n_hole_max]]
    if len(cand_idx) < 20:
        # Expand radius
        order = np.argsort(dists)
        cand_idx = order[:max(20, int(0.1 * n))]

    removed_flag = np.zeros(n, dtype=bool)
    removed_flag[cand_idx] = True

    # P = full cluster; Q = cluster minus hole
    P = cluster_vecs
    Q = cluster_vecs[~removed_flag]
    if len(Q) < 30:
        return {"auc": float("nan"), "n_removed": len(cand_idx), "n_cluster": n, "error": "Q too small"}

    # Reduce dimensionality
    if P.shape[1] > d_reduce:
        pca = PCA(n_components=d_reduce, random_state=42)
        combined = np.concatenate([P, Q])
        combined_r = pca.fit_transform(combined)
        P_r = combined_r[:len(P)]
        Q_r = combined_r[len(P):]
    else:
        P_r = P.copy()
        Q_r = Q.copy()

    # Run RuLSIF: estimate r(x) = p(x)/(α p(x) + (1-α) q(x))
    # Gaps correspond to LOW r (p has points but q doesn't there? No: p=full, q=no-hole.
    # So in hole region, p has points but q doesn't → p(x) > q(x) → r(x) large.
    # AUC of r_α as "is this point removed" should be HIGH (removed ↔ high r).
    try:
        model = run_rulsif(P_r, Q_r, alpha=alpha, kernel_num=kernel_num)
    except Exception as e:
        return {"auc": float("nan"), "n_removed": len(cand_idx), "n_cluster": n, "error": str(e)}

    r = model.compute_density_ratio(P_r)
    try:
        auc = roc_auc_score(removed_flag.astype(int), r)
    except Exception:
        auc = float("nan")

    return {
        "auc": float(auc),
        "n_removed": int(removed_flag.sum()),
        "n_cluster": n,
        "n_q": len(Q),
        "alpha": alpha,
        "kernel_num": kernel_num,
        "d_reduce": d_reduce,
        "hole_radius": hole_radius,
        "r_range": [float(r.min()), float(r.max())],
        "r_mean_removed": float(r[removed_flag].mean()),
        "r_mean_kept": float(r[~removed_flag].mean()),
    }


def permutation_null(
    X_p: np.ndarray,
    X_q: np.ndarray,
    alpha: float = 0.1,
    kernel_num: int = 100,
    n_perms: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Permutation null distribution of max |r_α - 1|.

    Shuffle labels p/q, fit RuLSIF on the shuffle, record extremes.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    combined = np.concatenate([X_p, X_q])
    n_p = len(X_p)
    n_total = len(combined)
    extremes = []
    for _ in range(n_perms):
        perm = rng.permutation(n_total)
        Xp_s = combined[perm[:n_p]]
        Xq_s = combined[perm[n_p:]]
        try:
            m = run_rulsif(Xp_s, Xq_s, alpha=alpha, kernel_num=kernel_num)
            r = m.compute_density_ratio(combined)
            extremes.append(float(np.abs(r - 1.0).max()))
        except Exception:
            continue
    extremes = np.array(extremes)
    return {
        "n_perms_ok": len(extremes),
        "mean_extreme": float(extremes.mean()) if len(extremes) else float("nan"),
        "p95_extreme": float(np.percentile(extremes, 95)) if len(extremes) else float("nan"),
    }
