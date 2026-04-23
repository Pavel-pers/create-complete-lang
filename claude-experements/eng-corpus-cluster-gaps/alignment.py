"""
Cluster-pair alignment: PCA → Gromov-Wasserstein → Procrustes → CCA-whitening.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sk_normalize


def center_and_reduce(
    X: np.ndarray,
    Y: np.ndarray,
    d_reduce: int = 30,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Concatenate X,Y → fit PCA on both → return reduced copies and fitted PCA.

    X, Y assumed centered per-cluster (mean subtracted before call).
    """
    n_x = len(X)
    both = np.concatenate([X, Y], axis=0)
    pca = PCA(n_components=d_reduce, random_state=42)
    both_r = pca.fit_transform(both)
    X_r = both_r[:n_x]
    Y_r = both_r[n_x:]
    return X_r.astype(np.float32), Y_r.astype(np.float32), pca


def gromov_wasserstein_coupling(
    X: np.ndarray,
    Y: np.ndarray,
    entropic_reg: float = 1e-3,
    max_iter: int = 500,
    max_points: int = 800,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute entropic Gromov-Wasserstein coupling between X and Y.

    To keep GW tractable we subsample each cloud to max_points.
    Returns (T coupling (nx, ny), sub_idx_x, sub_idx_y, info).
    """
    import ot

    if rng is None:
        rng = np.random.default_rng(42)

    n_x, n_y = len(X), len(Y)
    if n_x > max_points:
        idx_x = rng.choice(n_x, max_points, replace=False)
    else:
        idx_x = np.arange(n_x)
    if n_y > max_points:
        idx_y = rng.choice(n_y, max_points, replace=False)
    else:
        idx_y = np.arange(n_y)

    Xs = X[idx_x]
    Ys = Y[idx_y]

    Cx = ot.dist(Xs, Xs, metric="euclidean")
    Cy = ot.dist(Ys, Ys, metric="euclidean")
    Cx /= Cx.max() + 1e-12
    Cy /= Cy.max() + 1e-12

    p = np.ones(len(Xs)) / len(Xs)
    q = np.ones(len(Ys)) / len(Ys)

    T, log = ot.gromov.entropic_gromov_wasserstein(
        Cx, Cy, p, q,
        loss_fun="square_loss",
        epsilon=entropic_reg,
        max_iter=max_iter,
        tol=1e-9,
        log=True,
    )
    info = {
        "gw_distance": float(log.get("gw_dist", 0.0)),
        "n_x_sub": len(Xs),
        "n_y_sub": len(Ys),
    }
    return T, idx_x, idx_y, info


def mutual_nn_anchors(T: np.ndarray, top_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Extract mutual-argmax anchor pairs from GW coupling.

    For each row i, j* = argmax T[i]; for each column j, i* = argmax T[:,j].
    Anchor pair if j* == j and i* == i (mutual).
    Then keep top_ratio fraction by T[i,j*].
    """
    row_argmax = T.argmax(axis=1)
    col_argmax = T.argmax(axis=0)

    anchor_pairs = []
    for i, j in enumerate(row_argmax):
        if col_argmax[j] == i:
            anchor_pairs.append((i, j, T[i, j]))

    if not anchor_pairs:
        return np.array([], dtype=int), np.array([], dtype=int)

    anchor_pairs.sort(key=lambda x: -x[2])
    k = max(1, int(len(anchor_pairs) * top_ratio))
    picked = anchor_pairs[:k]
    return np.array([p[0] for p in picked]), np.array([p[1] for p in picked])


def procrustes_align(
    X_anchors: np.ndarray,
    Y_anchors: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orthogonal Procrustes: X @ R ≈ Y. Returns (R, residual_fraction).

    Residual fraction = ‖X·R − Y‖_F / ‖Y‖_F.
    """
    R, scale = orthogonal_procrustes(X_anchors, Y_anchors)
    # orthogonal_procrustes returns R s.t. X @ R best-matches Y (up to scale)
    residual = np.linalg.norm(X_anchors @ R - Y_anchors, "fro")
    norm_y = np.linalg.norm(Y_anchors, "fro") + 1e-12
    return R.astype(np.float32), float(residual / norm_y)


def csls_mutual_nn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CSLS-based mutual NN detection. Returns indices of matched pairs and frac.

    CSLS(x,y) = 2·cos(x,y) − r_x − r_y, where r_x = mean k-NN cos of x in Y.
    """
    Xn = sk_normalize(X, norm="l2", axis=1)
    Yn = sk_normalize(Y, norm="l2", axis=1)
    sim = Xn @ Yn.T  # (n_x, n_y)

    # r_x: each x's mean top-k sim in Y
    topk_in_y = np.partition(sim, -k, axis=1)[:, -k:]
    r_x = topk_in_y.mean(axis=1)
    topk_in_x = np.partition(sim, -k, axis=0)[-k:, :]
    r_y = topk_in_x.mean(axis=0)

    csls = 2 * sim - r_x[:, None] - r_y[None, :]

    # Mutual NN under CSLS
    row_argmax = csls.argmax(axis=1)
    col_argmax = csls.argmax(axis=0)
    x_idx, y_idx = [], []
    for i, j in enumerate(row_argmax):
        if col_argmax[j] == i:
            x_idx.append(i)
            y_idx.append(int(j))

    frac = len(x_idx) / max(min(len(X), len(Y)), 1)
    return np.array(x_idx), np.array(y_idx), frac


def cca_whitening(
    X: np.ndarray,
    Y: np.ndarray,
    reg: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """ZCA-style whitening: W_X = Cov(X)^(-1/2), W_Y = Cov(Y)^(-1/2).

    After whitening both clouds have ~identity covariance → single σ in RuLSIF works.
    """
    def zca(Z: np.ndarray) -> np.ndarray:
        Zc = Z - Z.mean(axis=0, keepdims=True)
        cov = (Zc.T @ Zc) / (len(Zc) - 1)
        U, s, Vt = np.linalg.svd(cov)
        S = np.diag(1.0 / np.sqrt(s + reg))
        W = (U @ S @ U.T).astype(np.float32)
        return Zc @ W

    return zca(X), zca(Y)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Squared MMD with RBF kernel between two samples (biased estimator)."""
    from scipy.spatial.distance import cdist

    if gamma is None:
        # median heuristic on combined sample
        n_max = 500
        rng = np.random.default_rng(42)
        xa = X[rng.choice(len(X), min(len(X), n_max), replace=False)]
        ya = Y[rng.choice(len(Y), min(len(Y), n_max), replace=False)]
        combined = np.concatenate([xa, ya])
        d = cdist(combined, combined)
        med = np.median(d[d > 0])
        gamma = 1.0 / (2 * med ** 2 + 1e-12)

    def k(a, b):
        d2 = cdist(a, b, metric="sqeuclidean")
        return np.exp(-gamma * d2)

    # Use random subsamples for tractability
    n_max = 800
    rng = np.random.default_rng(42)
    Xs = X[rng.choice(len(X), min(len(X), n_max), replace=False)]
    Ys = Y[rng.choice(len(Y), min(len(Y), n_max), replace=False)]

    Kxx = k(Xs, Xs)
    Kyy = k(Ys, Ys)
    Kxy = k(Xs, Ys)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def full_align(
    X: np.ndarray,
    Y: np.ndarray,
    d_reduce: int = 30,
    gw_max_points: int = 600,
    refine_iters: int = 3,
    do_whiten: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Full alignment pipeline. Returns dict with all intermediate artefacts."""
    if rng is None:
        rng = np.random.default_rng(42)

    # 1. Per-cluster centering
    mu_X = X.mean(axis=0, keepdims=True)
    mu_Y = Y.mean(axis=0, keepdims=True)
    Xc = (X - mu_X).astype(np.float32)
    Yc = (Y - mu_Y).astype(np.float32)

    # 2. PCA joint
    X_r, Y_r, pca = center_and_reduce(Xc, Yc, d_reduce=d_reduce)

    # MMD before alignment
    mmd_before = mmd_rbf(X_r, Y_r)

    # 3. GW coupling
    T, idx_x, idx_y, gw_info = gromov_wasserstein_coupling(
        X_r, Y_r, entropic_reg=1e-3, max_points=gw_max_points, rng=rng,
    )

    # 4. Mutual NN anchors from T
    a_x, a_y = mutual_nn_anchors(T, top_ratio=0.3)
    initial_anchor_frac = len(a_x) / max(min(len(idx_x), len(idx_y)), 1)

    if len(a_x) < 3:
        # Fallback: no good anchors → return centering-only alignment
        return {
            "X_aligned": X_r,
            "Y_aligned": Y_r,
            "R": np.eye(X_r.shape[1], dtype=np.float32),
            "pca": pca,
            "mu_X": mu_X,
            "mu_Y": mu_Y,
            "initial_anchor_frac": initial_anchor_frac,
            "procrustes_residual": float("nan"),
            "final_mutual_nn_frac": float("nan"),
            "mmd_before": mmd_before,
            "mmd_after": mmd_before,
            "gw_info": gw_info,
            "whitened": False,
            "note": "insufficient GW anchors, identity rotation",
        }

    # 5. Procrustes on anchors
    X_anchors = X_r[idx_x[a_x]]
    Y_anchors = Y_r[idx_y[a_y]]
    R, residual = procrustes_align(X_anchors, Y_anchors)

    # 6. Iterative refinement with CSLS
    X_aligned = X_r @ R
    best_frac = initial_anchor_frac
    for it in range(refine_iters):
        a_x_new, a_y_new, frac = csls_mutual_nn(X_aligned, Y_r, k=10)
        if len(a_x_new) < 3:
            break
        R_new, residual_new = procrustes_align(X_r[a_x_new], Y_r[a_y_new])
        X_aligned = X_r @ R_new
        R = R_new
        residual = residual_new
        best_frac = frac

    # 7. Optional CCA whitening
    # NOTE: whitening per-cluster individually destroys the *local* density
    # difference we want to measure (it normalizes each cloud to identity covariance
    # separately). We therefore disable it by default; only useful as a diagnostic.
    whitened = False
    if do_whiten and residual > 0.15:
        X_aligned, Y_aligned = cca_whitening(X_aligned, Y_r)
        whitened = True
    else:
        Y_aligned = Y_r

    mmd_after = mmd_rbf(X_aligned, Y_aligned)

    return {
        "X_aligned": X_aligned,
        "Y_aligned": Y_aligned,
        "R": R,
        "pca": pca,
        "mu_X": mu_X,
        "mu_Y": mu_Y,
        "initial_anchor_frac": initial_anchor_frac,
        "procrustes_residual": residual,
        "final_mutual_nn_frac": best_frac,
        "mmd_before": mmd_before,
        "mmd_after": mmd_after,
        "gw_info": gw_info,
        "whitened": whitened,
    }
