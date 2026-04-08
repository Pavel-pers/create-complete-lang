"""
Clustering artifact I/O — append to src/cclang/io/artifacts.py
 
File layout on disk (local or S3):
 
    clustering/{run_id}/
        meta.json               ← ClusteringMeta (parameters, stats, provenance)
        labels.npz              ← per-word arrays (see below)
        hierarchy.npz           ← Wishart-only: full event log for replay
 
    density_ratio/{run_id}/
        meta.json               ← DensityRatioMeta
        scores.npy              ← shape (n_words,) — r_α per word
 
Arrays inside labels.npz
──────────────────────────
    cluster_ids : int32   (n_words,)   — cluster label (-1 = noise)
    densities   : float32 (n_words,)   — local density estimate (r_i for Wishart, core_dist for HDBSCAN)
    roles       : int8    (n_words,)   — 0=core, 1=border, 2=noise (DBSCAN/HDBSCAN; all 0 for Wishart)
 
Arrays inside hierarchy.npz  (Wishart only)
────────────────────────────────────────────
    event_radii   : float32 (n_events,)       — r at which event fires
    event_types   : int8    (n_events,)       — 0=ActivateVertex, 1=ActivateEdge
    event_payload : int32   (n_events, 2)     — (i, 0) for vertex, (i, j) for edge
    knn_radii     : float32 (n_words,)        — r_i (k-th neighbour distance per word)
 
    Replay: iterate events in order → DSU → recover C(r) for any r.
"""

from __future__ import annotations
 
import json
from pathlib import Path
from typing import Optional, Tuple
 
import numpy as np

def save_clustering_labels(
    dir_path: str | Path,
    cluster_ids: np.ndarray,
    densities: np.ndarray,
    roles: Optional[np.ndarray] = None,
) -> Path:
    """
    Save per-word clustering assignments.
 
    Parameters
    ----------
    cluster_ids : (n_words,) int32 — cluster label, -1 = noise
    densities   : (n_words,) float32 — local density estimate
    roles       : (n_words,) int8 — 0=core, 1=border, 2=noise (optional)
    """
    out = _ensure_dir(Path(dir_path) / "labels.npz")
    arrays = {
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int32),
        "densities": np.asarray(densities, dtype=np.float32),
    }
    if roles is not None:
        arrays["roles"] = np.asarray(roles, dtype=np.int8)
    else:
        arrays["roles"] = np.zeros(len(cluster_ids), dtype=np.int8)
    np.savez_compressed(out, **arrays)
    return out
 
 
def load_clustering_labels(dir_path: str | Path) -> dict[str, np.ndarray]:
    """
    Returns dict with keys: 'cluster_ids', 'densities', 'roles'.
    """
    data = np.load(Path(dir_path) / "labels.npz")
    return {k: data[k] for k in data.files}
 
 
# ──── Wishart hierarchy ────
 
def save_wishart_hierarchy(
    dir_path: str | Path,
    event_radii: np.ndarray,
    event_types: np.ndarray,
    event_payload: np.ndarray,
    knn_radii: np.ndarray,
) -> Path:
    """
    Save the full Wishart event log for hierarchy replay.
 
    Parameters
    ----------
    event_radii   : (n_events,) float32
    event_types   : (n_events,) int8    — 0 = ActivateVertex, 1 = ActivateEdge
    event_payload : (n_events, 2) int32 — (i, 0) for vertex, (i, j) for edge
    knn_radii     : (n_words,) float32  — k-th neighbour distance per word
    """
    out = _ensure_dir(Path(dir_path) / "hierarchy.npz")
    np.savez_compressed(
        out,
        event_radii=np.asarray(event_radii, dtype=np.float32),
        event_types=np.asarray(event_types, dtype=np.int8),
        event_payload=np.asarray(event_payload, dtype=np.int32),
        knn_radii=np.asarray(knn_radii, dtype=np.float32),
    )
    return out
 
 
def load_wishart_hierarchy(dir_path: str | Path) -> dict[str, np.ndarray]:
    """
    Returns dict: 'event_radii', 'event_types', 'event_payload', 'knn_radii'.
    """
    data = np.load(Path(dir_path) / "hierarchy.npz")
    return {k: data[k] for k in data.files}
 
 
def replay_wishart_at_level(hierarchy: dict[str, np.ndarray], r_cut: float) -> np.ndarray:
    """
    Replay Wishart events up to radius *r_cut* and return cluster labels.
 
    Parameters
    ----------
    hierarchy : output of load_wishart_hierarchy()
    r_cut     : radius at which to slice the hierarchy
 
    Returns
    -------
    labels : (n_words,) int32 — cluster id per word, -1 = not yet activated
    """
    radii = hierarchy["event_radii"]
    types = hierarchy["event_types"]
    payload = hierarchy["event_payload"]
    knn_r = hierarchy["knn_radii"]
 
    n_words = len(knn_r)
 
    # DSU
    parent = np.full(n_words, -1, dtype=np.int32)
    rank = np.zeros(n_words, dtype=np.int32)
    active = np.zeros(n_words, dtype=bool)
 
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
 
    mask = radii <= r_cut
    for idx in np.where(mask)[0]:
        t = types[idx]
        p = payload[idx]
        if t == 0:  # ActivateVertex
            i = p[0]
            if not active[i]:
                active[i] = True
                parent[i] = i
        elif t == 1:  # ActivateEdge
            i, j = p[0], p[1]
            if active[i] and active[j]:
                union(i, j)
 
    # Build labels: root of component = cluster id
    labels = np.full(n_words, -1, dtype=np.int32)
    for i in range(n_words):
        if active[i]:
            labels[i] = find(i)
 
    return labels
 
 
# ════════════════════════════════════════════════════════
#  Density ratio (RuLSIF)
# ════════════════════════════════════════════════════════
 
def save_density_ratio(
    dir_path: str | Path,
    meta: DensityRatioMeta,
    scores: np.ndarray,
    centers: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
) -> Tuple[Path, Path]:
    """
    Save density-ratio estimation results.
 
    Parameters
    ----------
    scores  : (n_words,) float32 — r_α value per word
    centers : (b, d) float32 — kernel centres (optional, for reuse)
    theta   : (b,) float32 — learned weights (optional, for reuse)
    """
    base = Path(dir_path)
    meta_path = _ensure_dir(base / "meta.json")
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
 
    scores_path = _ensure_dir(base / "scores.npy")
    np.save(scores_path, np.asarray(scores, dtype=np.float32))
 
    if centers is not None:
        np.save(_ensure_dir(base / "centers.npy"), np.asarray(centers, dtype=np.float32))
    if theta is not None:
        np.save(_ensure_dir(base / "theta.npy"), np.asarray(theta, dtype=np.float32))
 
    return meta_path, scores_path
 
 
def load_density_ratio(dir_path: str | Path) -> Tuple[DensityRatioMeta, np.ndarray]:
    base = Path(dir_path)
    meta = DensityRatioMeta.model_validate_json(
        (base / "meta.json").read_text(encoding="utf-8")
    )
    scores = np.load(base / "scores.npy")
    return meta, scores
 
 
def load_density_ratio_model(dir_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load kernel centres and theta for re-evaluation at new points."""
    base = Path(dir_path)
    centers = np.load(base / "centers.npy")
    theta = np.load(base / "theta.npy")
    return centers, theta