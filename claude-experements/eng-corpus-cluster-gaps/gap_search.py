"""
Gap candidate search in r_α field using:
1. k-NN local minima / maxima (depending on sign convention)
2. Level sets in k-NN graph
3. Persistent homology (H0 bars via gudhi)

Plus voting aggregation.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_local_extrema(
    coords: np.ndarray,
    r_values: np.ndarray,
    k: int = 15,
    quantile: float = 0.95,
    mode: str = "high",
) -> np.ndarray:
    """Points where r_values[i] is a local extremum among k neighbors.

    mode="high": r[i] >= r[neighbor] for all neighbors AND r[i] >= global quantile threshold.
    mode="low": r[i] <= r[neighbor] ... AND r[i] <= global (1-quantile) threshold.
    Returns indices of extrema.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, neigh = nn.kneighbors(coords)
    neigh = neigh[:, 1:]  # drop self

    if mode == "high":
        thresh = np.quantile(r_values, quantile)
        is_ext = np.all(r_values[:, None] >= r_values[neigh], axis=1) & (r_values >= thresh)
    elif mode == "low":
        thresh = np.quantile(r_values, 1 - quantile)
        is_ext = np.all(r_values[:, None] <= r_values[neigh], axis=1) & (r_values <= thresh)
    else:
        raise ValueError(mode)
    return np.where(is_ext)[0]


def level_set_components(
    coords: np.ndarray,
    r_values: np.ndarray,
    k: int = 10,
    quantile: float = 0.9,
    mode: str = "high",
) -> list[np.ndarray]:
    """Connected components of {x : r > threshold} in k-NN graph."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    if mode == "high":
        thresh = np.quantile(r_values, quantile)
        keep = r_values >= thresh
    elif mode == "low":
        thresh = np.quantile(r_values, 1 - quantile)
        keep = r_values <= thresh
    else:
        raise ValueError(mode)

    idx_keep = np.where(keep)[0]
    if len(idx_keep) < 2:
        return []

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, neigh = nn.kneighbors(coords[idx_keep])
    neigh = neigh[:, 1:]
    # Build sparse adjacency among kept points
    rows = []
    cols = []
    keep_set = set(idx_keep.tolist())
    idx_to_local = {g: i for i, g in enumerate(idx_keep)}
    for local_i, global_i in enumerate(idx_keep):
        for j in neigh[local_i]:
            if int(j) in keep_set:
                rows.append(local_i)
                cols.append(idx_to_local[int(j)])
    n = len(idx_keep)
    adj = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(n, n),
    )
    n_components, labels = connected_components(adj, directed=False)
    components = []
    for c in range(n_components):
        comp_local = np.where(labels == c)[0]
        if len(comp_local) >= 3:
            components.append(idx_keep[comp_local])
    return components


def persistent_homology_extrema(
    coords: np.ndarray,
    r_values: np.ndarray,
    mode: str = "high",
    n_top: int = 20,
) -> np.ndarray:
    """Identify points with longest H0 persistence in super-level filtration of r_values.

    Uses a simple 1D scalar-field persistence over a k-NN graph
    (built via gudhi.SimplexTree) and picks the n_top most persistent peaks.

    Returns indices of peak points.
    """
    import gudhi as gd
    from sklearn.neighbors import kneighbors_graph

    k = 10
    if mode == "low":
        values = -r_values
    else:
        values = r_values

    # Build k-NN graph as simplicial complex (1-skeleton only for H0)
    st = gd.SimplexTree()
    for i, v in enumerate(values):
        st.insert([i], filtration=-v)  # super-level via negation

    A = kneighbors_graph(coords, n_neighbors=k, mode="connectivity").tolil()
    for i in range(A.shape[0]):
        for j in A.rows[i]:
            if i < j:
                st.insert([i, int(j)], filtration=-max(values[i], values[j]))

    st.compute_persistence(persistence_dim_max=False)
    pairs = st.persistence_pairs()

    # Each pair: (birth_simplex, death_simplex). For H0, birth is a vertex.
    peaks = []
    for birth, death in pairs:
        if len(birth) == 1:
            if death:
                # Death filtration − birth filtration
                b_f = st.filtration(birth)
                d_f = st.filtration(death)
                persistence = d_f - b_f
                peaks.append((birth[0], persistence))
            else:
                peaks.append((birth[0], float("inf")))

    peaks.sort(key=lambda x: -x[1])
    return np.array([p[0] for p in peaks[:n_top]], dtype=int)


def vote_aggregate(candidate_sets: list[np.ndarray], min_votes: int = 2) -> np.ndarray:
    """Return indices appearing in ≥ min_votes of the candidate_sets."""
    from collections import Counter

    votes: Counter[int] = Counter()
    for cs in candidate_sets:
        for idx in cs:
            votes[int(idx)] += 1
    return np.array([i for i, c in votes.items() if c >= min_votes], dtype=int)


def find_gap_candidates(
    coords: np.ndarray,
    r_values: np.ndarray,
    k: int = 15,
    quantile_knn: float = 0.95,
    quantile_level: float = 0.9,
    mode: str = "high",
    use_persistence: bool = True,
    n_persistence: int = 30,
) -> dict:
    """Run all three detectors and aggregate by voting.

    mode="high" picks high r_α regions (numerator denser than denominator)
    = gaps in the *reference/denominator* cloud.
    """
    knn_idx = knn_local_extrema(coords, r_values, k=k, quantile=quantile_knn, mode=mode)
    level_comps = level_set_components(coords, r_values, k=10, quantile=quantile_level, mode=mode)
    level_reps = []
    for comp in level_comps:
        # Use most extreme point in each component as the representative
        if mode == "high":
            rep = comp[np.argmax(r_values[comp])]
        else:
            rep = comp[np.argmin(r_values[comp])]
        level_reps.append(int(rep))
    level_arr = np.array(level_reps, dtype=int)

    detectors = {"knn": knn_idx, "level": level_arr}
    if use_persistence:
        try:
            ph_idx = persistent_homology_extrema(coords, r_values, mode=mode, n_top=n_persistence)
            detectors["persistence"] = ph_idx
        except Exception as e:
            detectors["persistence_error"] = str(e)

    candidate_lists = [v for k_, v in detectors.items() if isinstance(v, np.ndarray)]
    voted = vote_aggregate(candidate_lists, min_votes=2) if len(candidate_lists) >= 2 else candidate_lists[0]

    return {
        "detectors": detectors,
        "candidates": voted,
        "n_candidates": len(voted),
    }
