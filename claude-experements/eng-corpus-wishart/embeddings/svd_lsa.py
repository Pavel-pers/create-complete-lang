"""
SVD-LSA embedding method.

Supports three TDM weighting schemes: log-entropy, tf-idf, PPMI.
Pipeline: documents → fragments → sparse TDM → SVD → word vectors.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.utils.extmath import randomized_svd


# ── Weighting schemes ────────────────────────────────────────────────────────

def log_entropy_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    local = tf_matrix.copy()
    local.data = np.log1p(local.data)

    n_docs = tf_matrix.shape[0]
    csc = tf_matrix.tocsc()
    col_sums = np.array(csc.sum(axis=0)).flatten()

    col_indices = np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr))
    p = csc.data / col_sums[col_indices]

    p_log_p = p * np.log(p + 1e-30)
    entropy = np.zeros(csc.shape[1])
    np.add.at(entropy, col_indices, -p_log_p)

    global_weights = 1.0 - entropy / np.log(n_docs + 1)
    return local.multiply(global_weights)


def tfidf_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    local = tf_matrix.copy()
    local.data = np.log1p(local.data)

    n_docs = tf_matrix.shape[0]
    csc = tf_matrix.tocsc()
    df = np.diff(csc.indptr)
    idf = np.log(n_docs / (df + 1))
    return local.multiply(idf)


def ppmi_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Positive Pointwise Mutual Information weighting."""
    csc = tf_matrix.tocsc().astype(np.float64)
    total = csc.sum()

    row_sums = np.array(csc.sum(axis=1)).flatten()
    col_sums = np.array(csc.sum(axis=0)).flatten()

    # PMI = log( P(w,d) / (P(w) * P(d)) ) = log( count * total / (row_sum * col_sum) )
    coo = csc.tocoo()
    expected = (row_sums[coo.row] * col_sums[coo.col]) / total
    pmi = np.log(coo.data * total / (expected + 1e-30) + 1e-30)
    pmi = np.maximum(pmi, 0)  # PPMI: clamp negative values

    result = sparse.csr_matrix((pmi, (coo.row, coo.col)), shape=csc.shape)
    return result


WEIGHTING = {
    "log-entropy": log_entropy_weighting,
    "tf-idf": tfidf_weighting,
    "ppmi": ppmi_weighting,
}


# ── Fragment building ─────────────────────────────────────────────────────────

def build_fragments(
    documents: list[list[list[str]]],
    word2idx: dict[str, int],
    fragment_size: int = 10,
    min_fragment_ratio: float = 0.5,
) -> tuple[list[list[int]], list[str]]:
    """Split documents into fragments of fragment_size sentences."""
    fragments: list[list[int]] = []
    fragment_ids: list[str] = []
    min_sents = max(1, int(fragment_size * min_fragment_ratio))

    for doc_idx, doc in enumerate(documents):
        for start in range(0, len(doc), fragment_size):
            chunk = doc[start: start + fragment_size]
            if len(chunk) < min_sents:
                continue
            token_ids = [
                word2idx[tok]
                for sent in chunk
                for tok in sent
                if tok in word2idx
            ]
            if not token_ids:
                continue
            frag_hash = hashlib.sha256(
                f"{doc_idx}:{start}:{token_ids}".encode()
            ).hexdigest()[:16]
            fragments.append(token_ids)
            fragment_ids.append(frag_hash)

    print(f"[svd-lsa] fragments: {len(fragments)}")
    return fragments, fragment_ids


# ── TDM ───────────────────────────────────────────────────────────────────────

def build_tdm(
    fragments: list[list[int]],
    vocab_size: int,
    weighting: str,
) -> sparse.csr_matrix:
    """Build weighted Term-Document Matrix."""
    print(f"[svd-lsa] building TDM ({weighting}) ...")
    rows, cols, vals = [], [], []
    for i, frag in enumerate(fragments):
        for tid, cnt in Counter(frag).items():
            rows.append(i)
            cols.append(tid)
            vals.append(cnt)

    tf_matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(fragments), vocab_size),
        dtype=np.float64,
    )
    print(f"[svd-lsa] TDM shape: {tf_matrix.shape}, nnz: {tf_matrix.nnz}")

    result = WEIGHTING[weighting](tf_matrix)
    return result


# ── SVD ───────────────────────────────────────────────────────────────────────

def run_svd(
    tdm: sparse.csr_matrix,
    k: int = 300,
    n_iter: int = 5,
    oversampling: int = 10,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Truncated randomized SVD. Returns U, sigma, Vt, stats."""
    print(f"[svd-lsa] SVD (k={k}, n_iter={n_iter}) ...")
    t0 = time.monotonic()

    U, sigma, Vt = randomized_svd(
        tdm, n_components=k, n_iter=n_iter,
        n_oversamples=oversampling, random_state=random_state,
    )

    wall_time = time.monotonic() - t0
    print(f"[svd-lsa] SVD done in {wall_time:.1f}s  U={U.shape} σ={sigma.shape} Vt={Vt.shape}")

    forb_sq = float(np.sum(tdm.data ** 2))
    evr = sigma ** 2 / forb_sq if forb_sq > 0 else np.zeros_like(sigma)
    cum = np.cumsum(evr)

    stats = {
        "input_shape": list(tdm.shape),
        "input_nnz": int(tdm.nnz),
        "k": k,
        "n_iter": n_iter,
        "oversampling": oversampling,
        "random_state": random_state,
        "wall_time_sec": round(wall_time, 1),
        "energy_captured": round(float(cum[-1]) if len(cum) else 0, 4),
        "sigma_max": round(float(sigma.max()), 2),
        "sigma_min": round(float(sigma.min()), 2),
        "sigma_mean": round(float(sigma.mean()), 3),
    }

    return U, sigma, Vt, stats


# ── Embeddings ────────────────────────────────────────────────────────────────

def build_word_vectors(
    sigma: np.ndarray,
    Vt: np.ndarray,
    sigma_power: float = 0.5,
    reshape_k: int | None = None,
) -> np.ndarray:
    """Word vectors = Vt.T · diag(σ^α)."""
    if reshape_k is None:
        reshape_k = len(sigma)

    if sigma_power == 0.0:
        word_vectors = Vt.T[:, :reshape_k]
    elif sigma_power == 1.0:
        word_vectors = Vt.T[:, :reshape_k] * sigma[np.newaxis, :reshape_k]
    else:
        word_vectors = Vt.T[:, :reshape_k] * (sigma ** sigma_power)[np.newaxis, :reshape_k]

    return word_vectors


def postprocess(
    word_vectors: np.ndarray,
    center: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Center and/or L2-normalize embeddings."""
    result = word_vectors.copy()
    if center:
        result -= result.mean(axis=0)
    if normalize:
        result = sk_normalize(result, norm="l2", axis=1)
    return result


# ── Main build function ──────────────────────────────────────────────────────

def build(
    documents: list[list[list[str]]],
    word2idx: dict[str, int],
    vocab_size: int,
    weighting: str = "log-entropy",
    svd_k: int = 300,
    sigma_power: float = 0.5,
    fragment_size: int = 10,
    do_center: bool = True,
    do_normalize: bool = True,
    n_iter: int = 5,
    oversampling: int = 10,
    random_state: int | None = 42,
) -> tuple[np.ndarray, dict]:
    """
    Full SVD-LSA embedding pipeline.

    Returns (word_vectors, meta_dict).
    """
    t0 = time.monotonic()

    fragments, fragment_ids = build_fragments(documents, word2idx, fragment_size)
    tdm = build_tdm(fragments, vocab_size, weighting)
    del fragments, fragment_ids

    U, sigma, Vt, svd_stats = run_svd(tdm, svd_k, n_iter, oversampling, random_state)
    del U, tdm

    raw_vectors = build_word_vectors(sigma, Vt, sigma_power)
    del Vt

    word_vectors = postprocess(raw_vectors, center=do_center, normalize=do_normalize)

    norms = np.linalg.norm(word_vectors, axis=1)
    total_time = time.monotonic() - t0

    meta = {
        "method": "svd-lsa",
        "weighting": weighting,
        "svd_k": svd_k,
        "sigma_power": sigma_power,
        "fragment_size": fragment_size,
        "postprocess": {
            "center": do_center,
            "normalize": do_normalize,
        },
        "svd_stats": svd_stats,
        "embedding_stats": {
            "shape": list(word_vectors.shape),
            "norm_mean": round(float(norms.mean()), 6),
            "norm_std": round(float(norms.std()), 6),
            "norm_min": round(float(norms.min()), 6),
            "norm_max": round(float(norms.max()), 6),
            "zero_norm_count": int((norms == 0).sum()),
        },
        "total_time_sec": round(total_time, 1),
    }

    print(f"[svd-lsa] embeddings shape={word_vectors.shape}, "
          f"norm mean={norms.mean():.4f}, total={total_time:.1f}s")

    return word_vectors, meta
