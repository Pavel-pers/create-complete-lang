"""
Wishart density-based hierarchical clustering.

Algorithm:
1. Build KNN graph (k nearest neighbors per word)
2. Create event log:
   - ActivateVertex(r=knn_radii[i], i) for each word
   - ActivateEdge(r=dist(i,j), i, j) for each KNN edge
3. Sort events by radius
4. Replay events up to r_cut using DSU to get cluster labels

The full hierarchy is saved so clusters can be reconstructed at any r_cut.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn(
    vectors: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute k nearest neighbors for each vector.

    Returns (knn_indices, knn_distances, knn_radii).
    knn_radii[i] = distance to the k-th neighbor of point i.
    """
    print(f"[wishart] building KNN (k={k}, metric={metric}, n={vectors.shape[0]}) ...")
    t0 = time.monotonic()

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    # Remove self-neighbor (index 0)
    knn_distances = distances[:, 1:].astype(np.float32)
    knn_indices = indices[:, 1:].astype(np.int32)
    knn_radii = knn_distances[:, -1]  # k-th neighbor distance

    elapsed = time.monotonic() - t0
    print(f"[wishart] KNN done in {elapsed:.1f}s, "
          f"radii range: [{knn_radii.min():.4f}, {knn_radii.max():.4f}]")

    return knn_indices, knn_distances, knn_radii


def build_hierarchy(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    knn_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Wishart event hierarchy from KNN graph.

    Returns (event_radii, event_types, event_payload).
    """
    n_words = len(knn_radii)
    k = knn_indices.shape[1]

    print(f"[wishart] building hierarchy (n={n_words}, k={k}) ...")
    t0 = time.monotonic()

    # Vertex events
    vertex_radii = knn_radii.copy()
    vertex_types = np.zeros(n_words, dtype=np.int8)
    vertex_payload = np.column_stack([np.arange(n_words, dtype=np.int32),
                                      np.zeros(n_words, dtype=np.int32)])

    # Edge events — collect unique edges from KNN graph.
    # Edge weight = mutual reachability distance = max(dist(i,j), knn_radii[i], knn_radii[j])
    # This guarantees that when the edge event fires, both endpoints are already active.
    edge_set: set[tuple[int, int]] = set()
    edge_radii_list = []
    edge_payload_list = []

    for i in range(n_words):
        r_i = knn_radii[i]
        for j_idx in range(k):
            j = int(knn_indices[i, j_idx])
            edge_key = (min(i, j), max(i, j))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                raw_dist = knn_distances[i, j_idx]
                mutual_reach = max(raw_dist, r_i, knn_radii[j])
                edge_radii_list.append(mutual_reach)
                edge_payload_list.append(edge_key)

    n_edges = len(edge_radii_list)
    print(f"[wishart] unique edges: {n_edges}")

    edge_radii = np.array(edge_radii_list, dtype=np.float32)
    edge_types = np.ones(n_edges, dtype=np.int8)
    edge_payload = np.array(edge_payload_list, dtype=np.int32)

    # Merge and sort
    all_radii = np.concatenate([vertex_radii, edge_radii])
    all_types = np.concatenate([vertex_types, edge_types])
    all_payload = np.concatenate([vertex_payload, edge_payload], axis=0)

    # Sort by radius, vertex events before edge events at same radius
    sort_key = np.lexsort((all_types, all_radii))
    event_radii = all_radii[sort_key]
    event_types = all_types[sort_key]
    event_payload = all_payload[sort_key]

    elapsed = time.monotonic() - t0
    print(f"[wishart] hierarchy built in {elapsed:.1f}s, "
          f"total events: {len(event_radii)}")

    return event_radii, event_types, event_payload


def replay_at_level(
    event_radii: np.ndarray,
    event_types: np.ndarray,
    event_payload: np.ndarray,
    knn_radii: np.ndarray,
    r_cut: float,
) -> np.ndarray:
    """
    Replay Wishart events up to radius r_cut and return cluster labels.

    Returns labels: (n_words,) int32, -1 = not yet activated (noise).
    """
    n_words = len(knn_radii)

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

    mask = event_radii <= r_cut
    for idx in np.where(mask)[0]:
        t = event_types[idx]
        p = event_payload[idx]
        if t == 0:  # ActivateVertex
            i = p[0]
            if not active[i]:
                active[i] = True
                parent[i] = i
        elif t == 1:  # ActivateEdge
            i, j = p[0], p[1]
            if active[i] and active[j]:
                union(i, j)

    labels = np.full(n_words, -1, dtype=np.int32)
    for i in range(n_words):
        if active[i]:
            labels[i] = find(i)

    return labels


def sweep_cluster_counts(
    event_radii: np.ndarray,
    event_types: np.ndarray,
    event_payload: np.ndarray,
    knn_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep all events and record cluster count at each event.

    Returns (r_values, cluster_counts, active_counts) at each event.
    """
    n_words = len(knn_radii)
    n_events = len(event_radii)

    parent = np.full(n_words, -1, dtype=np.int32)
    rank = np.zeros(n_words, dtype=np.int32)
    active = np.zeros(n_words, dtype=bool)
    n_clusters = 0
    n_active = 0

    r_values = np.zeros(n_events, dtype=np.float32)
    cluster_counts = np.zeros(n_events, dtype=np.int32)
    active_counts = np.zeros(n_events, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idx in range(n_events):
        t = event_types[idx]
        p = event_payload[idx]

        if t == 0:  # ActivateVertex
            i = p[0]
            if not active[i]:
                active[i] = True
                parent[i] = i
                n_clusters += 1
                n_active += 1
        elif t == 1:  # ActivateEdge
            i, j = p[0], p[1]
            if active[i] and active[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    if rank[ri] < rank[rj]:
                        ri, rj = rj, ri
                    parent[rj] = ri
                    if rank[ri] == rank[rj]:
                        rank[ri] += 1
                    n_clusters -= 1

        r_values[idx] = event_radii[idx]
        cluster_counts[idx] = n_clusters
        active_counts[idx] = n_active

    return r_values, cluster_counts, active_counts


def find_r_cut_for_target(
    r_values: np.ndarray,
    cluster_counts: np.ndarray,
    target_min: int,
    target_max: int,
) -> list[float]:
    """
    Find r_cut values that give cluster counts in [target_min, target_max].
    Returns list of candidate r_cut values (event radii where count is in range).
    """
    mask = (cluster_counts >= target_min) & (cluster_counts <= target_max)
    candidates = r_values[mask]
    if len(candidates) == 0:
        return []
    return candidates.tolist()


def assign_noise_to_nearest(
    labels: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    """Assign noise points (label=-1) to nearest cluster centroid."""
    result = labels.copy()
    noise_mask = result == -1
    if not noise_mask.any():
        return result

    unique_labels = np.unique(result[~noise_mask])
    if len(unique_labels) == 0:
        return result

    # Compute centroids
    centroids = np.zeros((len(unique_labels), vectors.shape[1]))
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    for lab in unique_labels:
        centroids[label_map[lab]] = vectors[result == lab].mean(axis=0)

    # Assign noise to nearest centroid
    noise_indices = np.where(noise_mask)[0]
    noise_vecs = vectors[noise_indices]
    dists = np.linalg.norm(noise_vecs[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    for i, ni in enumerate(noise_indices):
        result[ni] = unique_labels[nearest[i]]

    return result


def merge_small_clusters(
    labels: np.ndarray,
    vectors: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """Merge clusters smaller than min_size into nearest larger cluster."""
    result = labels.copy()

    while True:
        unique, counts = np.unique(result[result >= 0], return_counts=True)
        small_mask = counts < min_size
        if not small_mask.any():
            break

        large_labels = unique[~small_mask]
        if len(large_labels) == 0:
            break

        # Compute centroids of large clusters
        large_centroids = np.zeros((len(large_labels), vectors.shape[1]))
        for idx, lab in enumerate(large_labels):
            large_centroids[idx] = vectors[result == lab].mean(axis=0)

        # Merge each small cluster into nearest large cluster
        merged_any = False
        for lab in unique[small_mask]:
            small_vecs = vectors[result == lab]
            small_centroid = small_vecs.mean(axis=0)
            dists = np.linalg.norm(large_centroids - small_centroid, axis=1)
            nearest_large = large_labels[np.argmin(dists)]
            result[result == lab] = nearest_large
            merged_any = True

        if not merged_any:
            break

    return result


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters to sequential 0..N-1 (noise stays -1)."""
    result = labels.copy()
    unique = np.unique(result[result >= 0])
    mapping = {old: new for new, old in enumerate(unique)}
    for i in range(len(result)):
        if result[i] >= 0:
            result[i] = mapping[result[i]]
    return result


# ── Main build function ──────────────────────────────────────────────────────

def build(
    vectors: np.ndarray,
    knn_k: int = 15,
    target_clusters: int | None = None,
    r_cut: float | None = None,
    min_cluster_size: int = 100,
    merge_small: bool = True,
    assign_noise: bool = True,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Full Wishart clustering pipeline.

    Either target_clusters or r_cut must be specified.

    Returns (labels, densities, meta_dict).
    """
    t0 = time.monotonic()

    knn_indices, knn_distances, knn_radii = build_knn(vectors, knn_k, metric)
    event_radii, event_types, event_payload = build_hierarchy(
        knn_indices, knn_distances, knn_radii
    )

    if r_cut is not None:
        chosen_r_cut = r_cut
    elif target_clusters is not None:
        # Sweep to find best r_cut
        r_vals, c_counts, a_counts = sweep_cluster_counts(
            event_radii, event_types, event_payload, knn_radii
        )
        # Find r_cut giving closest to target_clusters
        target_range_min = max(1, int(target_clusters * 0.8))
        target_range_max = int(target_clusters * 1.2)
        candidates = find_r_cut_for_target(r_vals, c_counts, target_range_min, target_range_max)
        if not candidates:
            # Fallback: find r_cut closest to target
            diffs = np.abs(c_counts - target_clusters)
            best_idx = np.argmin(diffs)
            chosen_r_cut = float(r_vals[best_idx])
            print(f"[wishart] no exact match; closest: {c_counts[best_idx]} clusters at r_cut={chosen_r_cut:.4f}")
        else:
            # Pick middle candidate
            chosen_r_cut = candidates[len(candidates) // 2]
    else:
        raise ValueError("Either target_clusters or r_cut must be specified")

    labels = replay_at_level(event_radii, event_types, event_payload, knn_radii, chosen_r_cut)

    n_noise = int((labels == -1).sum())
    n_clusters_before = len(np.unique(labels[labels >= 0]))

    if assign_noise and n_noise > 0:
        labels = assign_noise_to_nearest(labels, vectors)

    if merge_small:
        labels = merge_small_clusters(labels, vectors, min_cluster_size)

    labels = relabel_sequential(labels)

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    n_clusters = len(unique)

    elapsed = time.monotonic() - t0

    meta = {
        "method": "wishart",
        "knn_k": knn_k,
        "metric": metric,
        "r_cut": chosen_r_cut,
        "target_clusters": target_clusters,
        "min_cluster_size": min_cluster_size,
        "merge_small": merge_small,
        "assign_noise": assign_noise,
        "n_clusters": n_clusters,
        "n_clusters_before_merge": n_clusters_before,
        "n_noise_before_assign": n_noise,
        "cluster_sizes": {
            "min": int(counts.min()) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
            "mean": round(float(counts.mean()), 1) if len(counts) else 0,
            "std": round(float(counts.std()), 1) if len(counts) else 0,
        },
        "total_time_sec": round(elapsed, 1),
        "hierarchy_events": len(event_radii),
    }

    print(f"[wishart] {n_clusters} clusters, r_cut={chosen_r_cut:.4f}, "
          f"sizes: [{counts.min()}-{counts.max()}], time={elapsed:.1f}s")

    # Save hierarchy arrays in meta for optional persistence
    meta["_hierarchy"] = {
        "event_radii": event_radii,
        "event_types": event_types,
        "event_payload": event_payload,
        "knn_radii": knn_radii,
    }

    densities = knn_radii

    return labels, densities, meta
