#!/usr/bin/env python3
"""
Generator for cluster_explore.ipynb.

Run from claude-experements/eng-corpus-cluster-explore/:
    python build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
OUT = HERE / "cluster_explore.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


def build() -> nbf.NotebookNode:
    cells = []

    # ═══════════════════════════════════════════════════════════
    # Header
    # ═══════════════════════════════════════════════════════════
    cells.append(md("""# Cluster Exploration — English NewLit

Визуализация и статистики финальных кластеров из задачи `eng-corpus-wishart`.

**Входные данные:**
- Embeddings: `v4-nopc1` — 79,485 × 300, L2-normalized SVD-LSA с удалённым PC1
- Large: `c11-large` — 18 кластеров (Bisecting K-Means)
- Medium: `c11-medium` — 100 кластеров
- Small: `c17-small` — 517 кластеров

**Структура ноутбука:**
1. Setup
2. Large (18) — статистики + визуализация
3. Medium (100)
4. Small (517) — подмножество Top-20 + Random-20
5. Cross-level hierarchy — связи между уровнями
6. Summary

Под каждым графиком — markdown ячейка для выводов пользователя.
"""))

    # ═══════════════════════════════════════════════════════════
    # Section 0 — Setup
    # ═══════════════════════════════════════════════════════════
    cells.append(md("## 0. Setup"))

    cells.append(code("""# Imports
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Autoreload helpers
%load_ext autoreload
%autoreload 2

# Style
sns.set_theme(style="whitegrid", context="notebook")
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["font.size"] = 10

import helpers as H
print("Setup OK")
"""))

    cells.append(code("""# Paths and constants
EXP_DIR = Path(".").resolve()
WISHART_DIR = EXP_DIR.parent / "eng-corpus-wishart"
EMB_DIR = WISHART_DIR / "data" / "embeddings" / "v4-nopc1"
CLUSTERS_DIR = WISHART_DIR / "data" / "clusters"
CACHE_DIR = EXP_DIR / "cache"
FIG_DIR = EXP_DIR / "figures"
CACHE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

TOP_N = 20
RANDOM_N = 20
SILHOUETTE_SAMPLE_LARGE = 20_000
SILHOUETTE_SAMPLE_MEDIUM = 10_000
SILHOUETTE_SAMPLE_SMALL = 15_000
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

print(f"EMB_DIR: {EMB_DIR}")
print(f"CLUSTERS_DIR: {CLUSTERS_DIR}")
"""))

    cells.append(code("""# Load embeddings
vectors, idx2word, tf, df_arr = H.load_embeddings(EMB_DIR)
print(f"Vectors: shape={vectors.shape}, dtype={vectors.dtype}")
print(f"Vocab: {len(idx2word)} words")
print(f"TF range: [{tf.min()}, {tf.max()}]")
print(f"Norms: min={np.linalg.norm(vectors, axis=1).min():.4f}, "
      f"max={np.linalg.norm(vectors, axis=1).max():.4f} (should be ~1)")
"""))

    cells.append(code("""# Load all three cluster levels
LEVELS = {
    "large": H.load_level(CLUSTERS_DIR, "c11-large"),
    "medium": H.load_level(CLUSTERS_DIR, "c11-medium"),
    "small": H.load_level(CLUSTERS_DIR, "c17-small"),
}

for name, lvl in LEVELS.items():
    sizes = np.bincount(lvl["labels"][lvl["labels"] >= 0])
    print(f"{name:7s}: {lvl['n_clusters']:3d} clusters, "
          f"sizes [{sizes.min()}-{sizes.max()}], "
          f"mean={sizes.mean():.0f}")
"""))

    cells.append(code("""# Compute centroids for each level (used throughout the notebook)
CENTROIDS = {}
for name, lvl in LEVELS.items():
    c = H.compute_centroids(vectors, lvl["labels"])
    CENTROIDS[name] = c
    np.save(CACHE_DIR / f"centroids_{name}.npy", c)
    print(f"{name}: centroids shape {c.shape}")
"""))

    cells.append(code("""# Run UMAP once on the full 79k embedding — reuse for all levels
umap_cache = CACHE_DIR / "umap_coords.npy"
umap_coords, umap_reducer = H.run_umap(
    vectors,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    cache_path=umap_cache,
)
print(f"UMAP coords: {umap_coords.shape}, "
      f"x range [{umap_coords[:,0].min():.2f}, {umap_coords[:,0].max():.2f}]")
"""))

    cells.append(code("""# Project centroids to UMAP space (for overlay on scatter plots)
CENTROIDS_2D = {}
for name, c in CENTROIDS.items():
    CENTROIDS_2D[name] = umap_reducer.transform(c)
    print(f"{name}: centroids 2D shape {CENTROIDS_2D[name].shape}")
"""))

    cells.append(code("""# Compute anchor words for each cluster in each level (used in labels)
ANCHORS = {}
for name, lvl in LEVELS.items():
    ANCHORS[name] = H.anchor_words(vectors, lvl["labels"], CENTROIDS[name], idx2word, top_k=5)
print("Anchors computed for all levels.")
print("Example (large, cluster 0):", ANCHORS["large"][0])
"""))

    # ═══════════════════════════════════════════════════════════
    # Section 1 — Large
    # ═══════════════════════════════════════════════════════════
    add_level_section(cells, "large", 18, "Large")

    # ═══════════════════════════════════════════════════════════
    # Section 2 — Medium
    # ═══════════════════════════════════════════════════════════
    add_level_section(cells, "medium", 100, "Medium")

    # ═══════════════════════════════════════════════════════════
    # Section 3 — Small (with top-20 + random-20)
    # ═══════════════════════════════════════════════════════════
    add_level_section(cells, "small", 517, "Small", with_selection=True)

    # ═══════════════════════════════════════════════════════════
    # Section 4 — Cross-level hierarchy
    # ═══════════════════════════════════════════════════════════
    add_cross_level_section(cells)

    # ═══════════════════════════════════════════════════════════
    # Section 5 — Summary
    # ═══════════════════════════════════════════════════════════
    cells.append(md("## 5. Summary"))

    cells.append(code("""# Consolidated metrics table
summary_rows = []
for name in ["large", "medium", "small"]:
    lvl = LEVELS[name]
    sizes = np.bincount(lvl["labels"][lvl["labels"] >= 0])
    dist = H.distribution_summary(sizes)
    # Quality metrics were stored in global QUALITY dict
    q = QUALITY[name]
    summary_rows.append({
        "level": name,
        "n_clusters": dist["n_clusters"],
        "min": dist["min"],
        "max": dist["max"],
        "mean": round(dist["mean"], 1),
        "std": round(dist["std"], 1),
        "CV": round(dist["cv"], 3),
        "max/min": round(dist["max_min_ratio"], 2),
        "Gini": round(dist["gini"], 3),
        "H_norm": round(dist["entropy_norm"], 3),
        "silhouette": round(q["silhouette"], 4),
        "DB": round(q["davies_bouldin"], 3),
        "CH": round(q["calinski_harabasz"], 1),
        "Dunn": round(q["dunn"], 4),
    })

summary_df = pd.DataFrame(summary_rows).set_index("level")
summary_df
"""))

    cells.append(md("""### Артефакты

- Notebook: `cluster_explore.ipynb`
- Helpers: `helpers.py`
- Figures: `figures/*.png`
- Cache: `cache/umap_coords.npy`, `cache/centroids_*.npy`
- Interactive HTML: `figures/sankey.html`, `figures/treemap.html`, `figures/sunburst.html`
- Report: `claude-report.md`
- Iterations log: `versions/iterations.md`
"""))

    # Build notebook
    nb = new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def add_level_section(cells, name: str, n_clusters: int, title: str, with_selection: bool = False):
    """Add full analysis section for one level."""
    sec_num = {"large": 1, "medium": 2, "small": 3}[name]
    cells.append(md(f"## {sec_num}. {title} — {n_clusters} кластеров"))

    # ─── Stats ────────────────────────────────────────────────
    cells.append(md(f"### {sec_num}.1 Статистики и метрики качества"))

    cells.append(code(f"""# Per-cluster basic stats
level = LEVELS["{name}"]
labels = level["labels"]
densities = level["densities"]
centroids = CENTROIDS["{name}"]
n_clusters_{name} = level["n_clusters"]

sizes_{name} = np.bincount(labels[labels >= 0])
per_cluster_{name} = H.compute_per_cluster_stats(vectors, labels, centroids, densities)
print(f"{{n_clusters_{name}}} clusters computed; columns: {{list(per_cluster_{name}.columns)}}")
per_cluster_{name}.describe()
"""))

    cells.append(code(f"""# Distribution summary
dist_{name} = H.distribution_summary(sizes_{name})
print(json.dumps(dist_{name}, indent=2, default=float))
"""))

    sample_size = {"large": "SILHOUETTE_SAMPLE_LARGE",
                   "medium": "SILHOUETTE_SAMPLE_MEDIUM",
                   "small": "SILHOUETTE_SAMPLE_SMALL"}[name]

    cells.append(code(f"""# Quality metrics: silhouette (stratified), Davies-Bouldin, Calinski-Harabasz, Dunn
sil_cache = CACHE_DIR / "silhouette_{name}.npz"
if sil_cache.exists():
    cached = np.load(sil_cache)
    sil_mean_{name} = float(cached["mean"])
    sil_scores_{name} = cached["scores"]
    sil_idx_{name} = cached["indices"]
    print(f"Silhouette loaded from cache: mean={{sil_mean_{name}:.4f}}")
else:
    sil_mean_{name}, sil_scores_{name}, sil_idx_{name} = H.silhouette_sampled(
        vectors, labels,
        sample_size={sample_size},
        min_per_cluster=20,
        metric="cosine",
        rng=np.random.default_rng(RNG_SEED),
    )
    np.savez(sil_cache, mean=sil_mean_{name}, scores=sil_scores_{name}, indices=sil_idx_{name})
    print(f"Silhouette computed: mean={{sil_mean_{name}:.4f}} (stratified sample n={{len(sil_idx_{name})}})")

db_{name} = H.davies_bouldin_full(vectors, labels)
ch_{name} = H.calinski_harabasz_full(vectors, labels)
max_radii_{name} = per_cluster_{name}["max_dist"].values
dunn_{name} = H.dunn_index_centroid(centroids, max_radii_{name}, metric="cosine")

# Store globally
QUALITY.setdefault("all", {{}})
QUALITY["{name}"] = {{
    "silhouette": sil_mean_{name},
    "davies_bouldin": db_{name},
    "calinski_harabasz": ch_{name},
    "dunn": dunn_{name},
}}

print(f"Silhouette: {{sil_mean_{name}:.4f}}")
print(f"Davies-Bouldin: {{db_{name}:.4f}} (lower = better)")
print(f"Calinski-Harabasz: {{ch_{name}:.1f}} (higher = better)")
print(f"Dunn (centroid-based): {{dunn_{name}:.4f}} (higher = better)")
"""))

    cells.append(code(f"""# Per-cluster silhouette (from stratified sample)
sample_labels_{name} = labels[sil_idx_{name}]
per_cluster_sil_{name} = H.silhouette_per_cluster(sil_scores_{name}, sample_labels_{name})

# Add silhouette + anchors + mean_density (already in per_cluster) to summary table
summary_{name} = per_cluster_{name}.copy()
summary_{name}["silhouette"] = [per_cluster_sil_{name}.get(cid, np.nan) for cid in summary_{name}.index]
summary_{name}["anchors"] = [", ".join(ANCHORS["{name}"].get(cid, [])[:3]) for cid in summary_{name}.index]
summary_{name} = summary_{name}.sort_values("size", ascending=False)
summary_{name}.head(20)
"""))

    if with_selection:
        cells.append(code(f"""# Selection: Top-20 by size + Random-20 — used for all per-cluster plots in Small
SEL_{name.upper()} = H.pick_top_and_random(sizes_{name}, TOP_N, RANDOM_N, rng=np.random.default_rng(RNG_SEED))
print(f"Top-20 cluster IDs: {{SEL_{name.upper()}['top']}}")
print(f"Random-20 cluster IDs: {{SEL_{name.upper()}['random']}}")
"""))

    # ─── Size distribution ────────────────────────────────────
    cells.append(md(f"### {sec_num}.2 Распределение размеров кластеров"))

    # Plot: Sorted bar chart of sizes
    cells.append(code(f"""# Plot: sorted bar chart of cluster sizes
fig, ax = plt.subplots(figsize=(12, 5))
sorted_sizes = np.sort(sizes_{name})[::-1]
ax.bar(range(len(sorted_sizes)), sorted_sizes, color="steelblue", edgecolor="black", linewidth=0.3)
ax.axhline(sorted_sizes.mean(), color="red", linestyle="--", alpha=0.7, label=f"mean={{sorted_sizes.mean():.0f}}")
ax.axhline(np.median(sorted_sizes), color="orange", linestyle="--", alpha=0.7, label=f"median={{np.median(sorted_sizes):.0f}}")
ax.set_xlabel("Cluster rank (descending by size)")
ax.set_ylabel("Cluster size")
ax.set_title(f"{title}: cluster sizes sorted (n={{len(sorted_sizes)}})")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_01_sizes_sorted.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}1 — описание*: распределение размеров кластеров, отсортированных по убыванию."))

    # Plot: rank-size log-log
    cells.append(code(f"""# Plot: rank-size log-log with power law fit
from scipy.stats import linregress
fig, ax = plt.subplots(figsize=(8, 6))
ranks = np.arange(1, len(sizes_{name}) + 1)
sorted_sizes_desc = np.sort(sizes_{name})[::-1]
ax.loglog(ranks, sorted_sizes_desc, "o", markersize=5, color="steelblue", label="clusters")

# Power law fit: log(size) = a * log(rank) + b
log_rank = np.log(ranks)
log_size = np.log(sorted_sizes_desc)
slope, intercept, r_value, _, _ = linregress(log_rank, log_size)
fit_y = np.exp(intercept + slope * log_rank)
ax.loglog(ranks, fit_y, "r--", label=f"fit: size ∝ rank^{{{{{{slope:.2f}}}}}}, R²={{r_value**2:.3f}}")

ax.set_xlabel("Rank (log)")
ax.set_ylabel("Size (log)")
ax.set_title(f"{title}: rank-size log-log")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_02_rank_size_loglog.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}2*: log-log rank-size — наклон ~-1 соответствует Zipf."))

    # Plot: intra-cluster distance-to-centroid distribution (KDE)
    cells.append(code(f"""# Plot: per-cluster KDE of distances from each point to its cluster centroid
# NOTE: field `densities` in labels.npz actually stores distance-to-own-centroid
# (Euclidean on L2-normalized vectors). Not probability density.
fig, ax = plt.subplots(figsize=(12, 6))

# For large (18): all clusters; for medium/small: limit to top N for clarity
max_show = min(20, n_clusters_{name})
top_cids = np.argsort(-sizes_{name})[:max_show]
cmap = plt.cm.tab20(np.linspace(0, 1, max_show))

for i, cid in enumerate(top_cids):
    mask = labels == cid
    sns.kdeplot(densities[mask], ax=ax, color=cmap[i], alpha=0.6, linewidth=1.5,
                label=f"{{cid}} (n={{mask.sum()}})" if max_show <= 18 else None)

ax.set_xlabel("Distance to own cluster centroid (Euclidean, L2-normalized space)")
ax.set_ylabel("KDE density")
ax.set_title(f"{title}: intra-cluster distance-to-centroid distribution (top {{max_show}} clusters)", fontsize=12)
if max_show <= 18:
    ax.legend(ncol=3, fontsize=8, loc="upper right", title="cluster (n)")
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_03_intra_dist_kde.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}3*: KDE расстояний от каждой точки до **её собственного центроида**. Узкий пик слева = компактный кластер; широкий/смещённый вправо = рыхлый."))

    # Plot: intra-cluster distance boxplot
    cells.append(code(f"""# Plot: boxplot of intra-cluster distance to centroid
fig, ax = plt.subplots(figsize=(14, 6))

# For small: only show selection
if "{name}" == "small":
    show_cids = SEL_SMALL["combined"]
    sel_groups = SEL_SMALL["group"]
else:
    show_cids = list(range(n_clusters_{name}))
    sel_groups = None

box_data = []
box_labels = []
box_colors = []
for i, cid in enumerate(show_cids):
    mask = labels == cid
    if mask.any():
        dists = np.linalg.norm(vectors[mask] - centroids[cid], axis=1)
        box_data.append(dists)
        box_labels.append(str(cid))
        if sel_groups is not None:
            box_colors.append("steelblue" if sel_groups[i] == "top" else "coral")
        else:
            box_colors.append("steelblue")

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)

ax.set_xlabel("Cluster ID")
ax.set_ylabel("Distance to centroid")
ax.set_title(f"{title}: intra-cluster distance to centroid")
if sel_groups is not None:
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="steelblue", label="Top-20"), Patch(color="coral", label="Random-20")])
if len(show_cids) > 30:
    plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_04_intra_dist_boxplot.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}4*: компактность кластеров — распределение расстояний точек до центроида."))

    # ─── Quality ──────────────────────────────────────────────
    cells.append(md(f"### {sec_num}.3 Качество кластеризации"))

    # Silhouette plot
    cells.append(code(f"""# Silhouette plot (horizontal bars per point, grouped by cluster)
fig, ax = plt.subplots(figsize=(10, max(6, n_clusters_{name} * 0.15)))

# For small, limit to selection for readability
if "{name}" == "small":
    sel_cids = SEL_SMALL["combined"]
    mask_sel = np.isin(sample_labels_{name}, sel_cids)
    ss = sil_scores_{name}[mask_sel]
    sl = sample_labels_{name}[mask_sel]
    show_cids = sel_cids
    ax.set_title(f"{title}: silhouette plot (selection: Top-20 + Random-20, sample={{mask_sel.sum()}})")
else:
    ss = sil_scores_{name}
    sl = sample_labels_{name}
    show_cids = list(range(n_clusters_{name}))
    ax.set_title(f"{title}: silhouette plot (sample n={{len(ss)}})")

y_lower = 0
cmap = plt.cm.tab20(np.linspace(0, 1, len(show_cids)))
for i, cid in enumerate(show_cids):
    cluster_ss = np.sort(ss[sl == cid])
    if len(cluster_ss) == 0:
        continue
    y_upper = y_lower + len(cluster_ss)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_ss,
                     facecolor=cmap[i], edgecolor=cmap[i], alpha=0.7)
    ax.text(-0.02, y_lower + len(cluster_ss) / 2, str(cid), fontsize=7, ha="right", va="center")
    y_lower = y_upper + 5

ax.axvline(sil_mean_{name}, color="red", linestyle="--", label=f"mean={{sil_mean_{name}:.3f}}")
ax.set_xlabel("Silhouette value")
ax.set_ylabel("Clusters (ordered)")
ax.legend()
ax.set_yticks([])
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_05_silhouette_plot.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}5*: классический silhouette plot — ширина полосы = silhouette score каждой точки."))

    # Bar: mean silhouette per cluster
    cells.append(code(f"""# Bar chart: mean silhouette per cluster
fig, ax = plt.subplots(figsize=(12, max(5, n_clusters_{name} * 0.08)))

cids = sorted(per_cluster_sil_{name}.keys())
vals = [per_cluster_sil_{name}[c] for c in cids]
order = np.argsort(vals)
cids_sorted = [cids[i] for i in order]
vals_sorted = [vals[i] for i in order]

colors = ["green" if v > sil_mean_{name} else "crimson" for v in vals_sorted]
ax.barh(range(len(cids_sorted)), vals_sorted, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
ax.axvline(sil_mean_{name}, color="black", linestyle="--", label=f"mean={{sil_mean_{name}:.3f}}")
ax.set_yticks(range(len(cids_sorted)))
ax.set_yticklabels(cids_sorted, fontsize=7 if n_clusters_{name} > 50 else 9)
ax.set_xlabel("Mean silhouette")
ax.set_ylabel("Cluster ID")
ax.set_title(f"{title}: mean silhouette per cluster")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_06_silhouette_per_cluster.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}6*: средний silhouette каждого кластера. Красные — ниже среднего."))

    # Scatter: size vs mean distance
    cells.append(code(f"""# Scatter: size vs mean intra-distance, color=silhouette, marker size ∝ inertia
fig, ax = plt.subplots(figsize=(10, 7))
df_plot = summary_{name}.copy()
sizes_s = df_plot["size"].values
mean_dists = df_plot["mean_dist"].values
sils = df_plot["silhouette"].values
inertias = df_plot["inertia"].values

marker_sizes = 50 + 250 * (inertias - inertias.min()) / (inertias.max() - inertias.min() + 1e-9)
sc = ax.scatter(sizes_s, mean_dists, c=sils, s=marker_sizes,
                cmap="RdYlGn", alpha=0.8, edgecolor="black", linewidth=0.5)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Mean silhouette")
ax.set_xlabel("Cluster size")
ax.set_ylabel("Mean distance to centroid")
ax.set_title(f"{title}: cluster size vs compactness (color=silhouette, marker size ∝ inertia)")
if "{name}" == "large":
    for cid, row in df_plot.iterrows():
        ax.annotate(str(cid), (row["size"], row["mean_dist"]), fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_07_size_vs_compactness.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}7*: зависимость компактности от размера кластера."))

    # ─── Inter-cluster structure ──────────────────────────────
    cells.append(md(f"### {sec_num}.4 Межкластерная структура"))

    # Heatmap of centroid distances
    cells.append(code(f"""# Heatmap of centroid cosine distances, ordered by hierarchical clustering
dist_mat = H.centroid_distance_matrix(centroids, metric="cosine")
Z = sch.linkage(squareform(dist_mat, checks=False), method="average")
leaves = sch.leaves_list(Z)
dist_mat_ord = dist_mat[leaves][:, leaves]

fig, ax = plt.subplots(figsize=(max(12, min(n_clusters_{name} * 0.12, 16)), max(10, min(n_clusters_{name} * 0.11, 14))))
sns.heatmap(dist_mat_ord,
            annot=(n_clusters_{name} <= 20),
            fmt=".2f",
            cmap="viridis_r",
            xticklabels=leaves if n_clusters_{name} <= 100 else False,
            yticklabels=leaves if n_clusters_{name} <= 100 else False,
            ax=ax, cbar_kws={{"label": "Cosine distance"}})
ax.set_title(f"{title}: centroid distance heatmap (leaf-ordered, {{n_clusters_{name}}}×{{n_clusters_{name}}})", fontsize=13)
ax.set_xlabel("Cluster ID (dendrogram order)")
ax.set_ylabel("Cluster ID (dendrogram order)")
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_08_centroid_heatmap.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}8*: матрица косинусных расстояний между центроидами."))

    # Dendrogram
    cells.append(code(f"""# Dendrogram of centroids
fig, ax = plt.subplots(figsize=(max(14, n_clusters_{name} * 0.15), 7))
labels_dendro = [f"{{cid}}: {{'/'.join(ANCHORS['{name}'].get(cid, [])[:2])}}" for cid in range(n_clusters_{name})]

# For small, truncate
if n_clusters_{name} > 100:
    sch.dendrogram(Z, ax=ax, truncate_mode="lastp", p=50, leaf_font_size=8,
                   show_contracted=True)
    ax.set_title(f"{title}: centroid dendrogram (truncated to 50 leaves)")
else:
    sch.dendrogram(Z, ax=ax, labels=labels_dendro, leaf_font_size=8 if n_clusters_{name} <= 20 else 6,
                   leaf_rotation=90)
    ax.set_title(f"{title}: centroid dendrogram (average linkage, cosine)")

ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_09_dendrogram.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}9*: иерархия кластеров (average linkage) с anchor-словами в листах."))

    # Network graph of closest pairs
    cells.append(code(f"""# Force-directed graph of closest centroid pairs
import networkx as nx
n_edges_show = int(n_clusters_{name} * (n_clusters_{name} - 1) / 2 * 0.10)  # top 10%
mask_tri = np.triu(np.ones_like(dist_mat, dtype=bool), k=1)
pairs = [(i, j, dist_mat[i, j]) for i, j in zip(*np.where(mask_tri))]
pairs.sort(key=lambda x: x[2])
top_pairs = pairs[:max(n_edges_show, n_clusters_{name})]

G = nx.Graph()
for i in range(n_clusters_{name}):
    G.add_node(i, size=int(sizes_{name}[i]))
for i, j, d in top_pairs:
    G.add_edge(int(i), int(j), weight=1.0 / (d + 1e-6))

pos = nx.spring_layout(G, seed=42, k=0.8)
fig, ax = plt.subplots(figsize=(12, 10))
node_sizes = [G.nodes[n]["size"] / 3 for n in G.nodes]
edge_widths = [G.edges[e]["weight"] * 0.3 for e in G.edges]
nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.4, edge_color="gray")
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="steelblue",
                        edgecolors="black", linewidths=0.5, alpha=0.85)
if n_clusters_{name} <= 100:
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
ax.set_title(f"{title}: cluster proximity graph (top 10% closest pairs, node size ∝ cluster size)")
ax.axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_10_network.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}10*: граф близости — связи между семантически близкими кластерами."))

    # ─── 2D projection ────────────────────────────────────────
    cells.append(md(f"### {sec_num}.5 2D проекция UMAP"))

    cells.append(code(f"""# UMAP scatter colored by cluster
fig, ax = plt.subplots(figsize=(13, 11))
k = n_clusters_{name}

if k <= 20:
    palette = plt.cm.tab20(np.linspace(0, 1, k))
elif k <= 100:
    palette = sns.husl_palette(k, l=0.65, s=0.85)
else:
    palette = sns.husl_palette(k, l=0.6, s=0.85)

# Point rendering
point_alpha = 0.35 if k <= 20 else (0.3 if k <= 100 else 0.22)
point_size = 4 if k <= 20 else (2.5 if k <= 100 else 2)

for cid in range(k):
    mask = labels == cid
    color = palette[cid]
    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               c=[color], s=point_size, alpha=point_alpha, linewidth=0)

# Centroids and labels
cent_2d = CENTROIDS_2D["{name}"]

if k <= 20:
    # All centroids labeled with big text
    ax.scatter(cent_2d[:, 0], cent_2d[:, 1], c="white", s=380, marker="o",
               edgecolors="black", linewidths=2, zorder=10)
    for i in range(k):
        ax.text(cent_2d[i, 0], cent_2d[i, 1], str(i), fontsize=13, fontweight="bold",
                ha="center", va="center", color="black", zorder=11)
else:
    # For 100/517: small centroid dots, label only top 10
    ax.scatter(cent_2d[:, 0], cent_2d[:, 1], c="black", s=25, marker="o",
               edgecolors="white", linewidths=0.8, zorder=10)
    top10 = np.argsort(-sizes_{name})[:10]
    for i in top10:
        ax.annotate(f"{{i}}", cent_2d[i], fontsize=10, fontweight="bold",
                    color="white",
                    xytext=(4, 4), textcoords="offset points",
                    path_effects=[
                        __import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(
                            linewidth=3, foreground="black")
                    ])

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title(f"{title}: UMAP projection colored by cluster ({{k}} clusters)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_11_umap_clusters.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}11*: UMAP-проекция с цветовой кодировкой кластеров и центроидами."))

    cells.append(code(f"""# UMAP colored by per-point distance-to-own-centroid
# (field `densities` in labels.npz = Euclidean distance to own cluster centroid)
fig, ax = plt.subplots(figsize=(12, 10))
sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                c=densities, s=2, alpha=0.4, cmap="magma_r", linewidth=0)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Distance to own cluster centroid")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title(f"{title}: UMAP coloured by distance to own cluster centroid", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "{name}_12_umap_dist_to_centroid.png")
plt.show()
"""))
    cells.append(md(f"> *Plot {title[0]}12*: UMAP, раскраска по расстоянию **до собственного центроида кластера**. Тёмные точки = близко к центру своего кластера (ядро); светлые = на периферии, далеко от центра. Помогает увидеть «размытые» или «плотные» кластеры в пространстве."))


def add_cross_level_section(cells):
    cells.append(md("## 4. Cross-level hierarchy"))

    cells.append(md("### 4.1 Flow matrices и purity"))

    cells.append(code("""# Build flow matrices between levels
labels_L = LEVELS["large"]["labels"]
labels_M = LEVELS["medium"]["labels"]
labels_S = LEVELS["small"]["labels"]
k_L = LEVELS["large"]["n_clusters"]
k_M = LEVELS["medium"]["n_clusters"]
k_S = LEVELS["small"]["n_clusters"]

flow_LM = H.build_flow_matrix(labels_L, labels_M, k_L, k_M)
flow_MS = H.build_flow_matrix(labels_M, labels_S, k_M, k_S)
flow_LS = H.build_flow_matrix(labels_L, labels_S, k_L, k_S)

# Sanity: row sums = cluster sizes
assert flow_LM.sum(axis=1).tolist() == np.bincount(labels_L).tolist(), "row sum mismatch"
print(f"flow_LM shape: {flow_LM.shape}, total: {flow_LM.sum()}")
print(f"flow_MS shape: {flow_MS.shape}, total: {flow_MS.sum()}")
print(f"flow_LS shape: {flow_LS.shape}, total: {flow_LS.sum()}")
"""))

    cells.append(code("""# Purity: how cleanly Medium nests into Large, Small into Medium
purity_M_in_L = H.purity_from_flow(flow_LM, axis=0)  # per Medium column
purity_S_in_M = H.purity_from_flow(flow_MS, axis=0)  # per Small column
purity_S_in_L = H.purity_from_flow(flow_LS, axis=0)

print(f"Medium clusters: mean purity in Large = {purity_M_in_L.mean():.3f}, "
      f"median = {np.median(purity_M_in_L):.3f}")
print(f"  >0.9 pure: {(purity_M_in_L > 0.9).sum()}/{k_M}")
print(f"  >0.95 pure: {(purity_M_in_L > 0.95).sum()}/{k_M}")

print(f"Small clusters: mean purity in Medium = {purity_S_in_M.mean():.3f}, "
      f"median = {np.median(purity_S_in_M):.3f}")
print(f"  >0.9 pure: {(purity_S_in_M > 0.9).sum()}/{k_S}")

print(f"Small clusters: mean purity in Large = {purity_S_in_L.mean():.3f}")
"""))

    cells.append(md("### 4.2 Визуализация flow между уровнями"))

    # X1: heatmap flow_LM
    cells.append(code("""# Heatmap flow_LM — how Medium clusters distribute over Large
import matplotlib.colors as mcolors

# Normalize per Medium column: fraction of Medium cluster coming from each Large
flow_LM_norm = flow_LM / (flow_LM.sum(axis=0, keepdims=True) + 1e-9)

# Order Medium columns so dominant Large alternates along x (visual)
dominant_L = flow_LM_norm.argmax(axis=0)
col_order = np.argsort(dominant_L)

# Transpose for better aspect ratio (100 rows × 18 cols)
fig, ax = plt.subplots(figsize=(10, 16))
sns.heatmap(flow_LM_norm[:, col_order].T,
            cmap="YlOrRd", vmin=0, vmax=1,
            xticklabels=range(k_L), yticklabels=col_order,
            ax=ax, cbar_kws={"label": "Fraction of Medium cluster"})
ax.set_ylabel("Medium cluster ID (ordered by dominant Large)")
ax.set_xlabel("Large cluster ID")
ax.set_title("Flow Large→Medium (normalized per Medium)", fontsize=13)
plt.setp(ax.get_yticklabels(), fontsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR / "cross_01_flow_LM.png")
plt.show()
"""))
    cells.append(md("> *Plot X1*: каждый столбец = один Medium-кластер. Тёплые ячейки показывают, из каких Large он состоит. Большинство Medium ∈ ровно один Large = хорошая иерархия."))

    # X2: flow_MS heatmap
    cells.append(code("""# Heatmap flow_MS (100x517)
flow_MS_norm = flow_MS / (flow_MS.sum(axis=0, keepdims=True) + 1e-9)
dominant_M = flow_MS_norm.argmax(axis=0)
col_order_S = np.argsort(dominant_M)

fig, ax = plt.subplots(figsize=(22, 10))
sns.heatmap(flow_MS_norm[:, col_order_S],
            cmap="YlOrRd", vmin=0, vmax=1,
            xticklabels=False, yticklabels=range(0, k_M, 5),
            ax=ax, cbar_kws={"label": "Fraction of Small cluster"})
ax.set_xlabel("Small cluster ID (ordered by dominant Medium)")
ax.set_ylabel("Medium cluster ID")
ax.set_title("Flow Medium→Small (normalized per Small column, 517 columns)")
plt.tight_layout()
plt.savefig(FIG_DIR / "cross_02_flow_MS.png")
plt.show()
"""))
    cells.append(md("> *Plot X2*: вложенность Small → Medium. Диагональная структура = каждый Small целиком принадлежит одному Medium."))

    # X3: Plotly Sankey
    cells.append(code("""# Plotly Sankey: Large -> Medium -> Small
import plotly.graph_objects as go
import plotly.io as pio

# Filter flows below threshold to reduce visual clutter
threshold_LM = 30
threshold_MS = 20

# Build node list: 0..k_L-1 = Large, k_L..k_L+k_M-1 = Medium, then Small
node_labels = []
node_colors = []

# Color palette for Large clusters
L_colors = [f"hsl({h},70%,50%)" for h in np.linspace(0, 340, k_L)]

for i in range(k_L):
    top_w = ANCHORS["large"].get(i, ["?"])[0]
    node_labels.append(f"L{i}: {top_w}")
    node_colors.append(L_colors[i])

# Propagate Large color to Medium and Small based on dominant source
for j in range(k_M):
    dominant = flow_LM[:, j].argmax()
    node_labels.append(f"M{j}")
    node_colors.append(L_colors[dominant])

for kk in range(k_S):
    dominant_M_for_S = flow_MS[:, kk].argmax()
    dominant_L_for_S = flow_LM[:, dominant_M_for_S].argmax()
    node_labels.append(f"S{kk}")
    node_colors.append(L_colors[dominant_L_for_S])

# Build links
sources, targets, values, link_colors = [], [], [], []
for i in range(k_L):
    for j in range(k_M):
        if flow_LM[i, j] >= threshold_LM:
            sources.append(i)
            targets.append(k_L + j)
            values.append(int(flow_LM[i, j]))
            link_colors.append(L_colors[i].replace("50%)", "50%,0.4)").replace("hsl", "hsla"))

for j in range(k_M):
    for kk in range(k_S):
        if flow_MS[j, kk] >= threshold_MS:
            sources.append(k_L + j)
            targets.append(k_L + k_M + kk)
            values.append(int(flow_MS[j, kk]))
            dom = flow_LM[:, j].argmax()
            link_colors.append(L_colors[dom].replace("50%)", "50%,0.3)").replace("hsl", "hsla"))

total = flow_LM.sum() + flow_MS.sum()
kept = sum(values)
print(f"Links kept: {len(sources)}, value total: {kept} ({100*kept/total:.1f}% of flows)")

fig = go.Figure(data=[go.Sankey(
    node=dict(pad=10, thickness=15, line=dict(color="black", width=0.3),
              label=node_labels, color=node_colors),
    link=dict(source=sources, target=targets, value=values, color=link_colors),
)])
fig.update_layout(title_text="Cross-level hierarchy: Large → Medium → Small",
                  font_size=9, height=900, width=1400)
fig.write_html(FIG_DIR / "cross_03_sankey.html")
try:
    fig.write_image(FIG_DIR / "cross_03_sankey.png", scale=1.5)
except Exception as e:
    print(f"[warn] PNG export failed: {e}")
fig.show()
"""))
    cells.append(md("> *Plot X3 (Sankey)*: потоки слов через уровни Large → Medium → Small. Цвет наследуется от Large-источника."))

    # X4: Plotly Treemap
    cells.append(code("""# Plotly Treemap: hierarchical Large > Medium > Small
import plotly.express as px

# Build hierarchical dataframe
tree_rows = []
for i in range(k_L):
    L_label = f"L{i}: {ANCHORS['large'].get(i, ['?'])[0]}"
    for j in range(k_M):
        if flow_LM[i, j] == 0:
            continue
        M_label = f"M{j}"
        for kk in range(k_S):
            if flow_MS[j, kk] == 0:
                continue
            # Check if this S is also in L_i (majority rule)
            if flow_LS[i, kk] == 0:
                continue
            S_label = f"S{kk}"
            tree_rows.append({
                "Large": L_label,
                "Medium": M_label,
                "Small": S_label,
                "size": int(flow_LS[i, kk]),
            })

tree_df = pd.DataFrame(tree_rows)
print(f"Treemap rows: {len(tree_df)}")

fig = px.treemap(tree_df, path=["Large", "Medium", "Small"], values="size",
                 color="size", color_continuous_scale="Viridis",
                 title="Hierarchical treemap: Large ⊃ Medium ⊃ Small")
fig.update_layout(height=800, width=1400)
fig.write_html(FIG_DIR / "cross_04_treemap.html")
try:
    fig.write_image(FIG_DIR / "cross_04_treemap.png", scale=1.5)
except Exception as e:
    print(f"[warn] PNG export failed: {e}")
fig.show()
"""))
    cells.append(md("> *Plot X4 (Treemap)*: вложенная визуализация иерархии с пропорциональными площадями."))

    # X5: Sunburst
    cells.append(code("""# Plotly Sunburst: alternative view
fig = px.sunburst(tree_df, path=["Large", "Medium", "Small"], values="size",
                  color="size", color_continuous_scale="Viridis",
                  title="Sunburst: Large ⊃ Medium ⊃ Small")
fig.update_layout(height=800, width=900)
fig.write_html(FIG_DIR / "cross_05_sunburst.html")
try:
    fig.write_image(FIG_DIR / "cross_05_sunburst.png", scale=1.5)
except Exception as e:
    print(f"[warn] PNG export failed: {e}")
fig.show()
"""))
    cells.append(md("> *Plot X5 (Sunburst)*: круговая альтернатива treemap."))

    # X6: Stacked bar chart
    cells.append(code("""# Stacked bar: for each Large cluster, Medium children
fig, ax = plt.subplots(figsize=(14, 7))

# For each Large, order Medium children by size, use husl colors
large_order = np.argsort(-np.bincount(labels_L))
x_pos = np.arange(k_L)

bottoms = np.zeros(k_L)
for j in range(k_M):
    vals = flow_LM[:, j]
    if vals.sum() == 0:
        continue
    ax.bar(x_pos, vals[large_order], bottom=bottoms[large_order], width=0.85,
           edgecolor="white", linewidth=0.2)
    bottoms[large_order] += vals[large_order]

ax.set_xlabel("Large cluster ID (sorted by size)")
ax.set_ylabel("Number of words")
ax.set_title("Each Large cluster stacked by Medium children")
ax.set_xticks(x_pos)
ax.set_xticklabels(large_order)
plt.tight_layout()
plt.savefig(FIG_DIR / "cross_06_stacked_bar.png")
plt.show()
"""))
    cells.append(md("> *Plot X6*: каждый Large-кластер разбит на Medium-дети — видно, как именно происходит дробление."))

    # X7: UMAP triptych
    cells.append(code("""# UMAP triptych: colored by Large / Medium / Small
fig, axes = plt.subplots(1, 3, figsize=(24, 9))

for ax, level_name in zip(axes, ["large", "medium", "small"]):
    lvl = LEVELS[level_name]
    k = lvl["n_clusters"]
    labs = lvl["labels"]
    if k <= 20:
        palette = plt.cm.tab20(np.linspace(0, 1, k))
    else:
        palette = np.array(sns.husl_palette(k, l=0.6, s=0.85))
    colors = palette[labs]
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors,
               s=2.5, alpha=0.35, linewidth=0)
    ax.set_title(f"{level_name.title()} ({k} clusters)", fontsize=15)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)

plt.suptitle("UMAP — same 2D coordinates, coloured by 3 hierarchy levels", fontsize=17, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "cross_07_umap_triptych.png")
plt.show()
"""))
    cells.append(md("> *Plot X7*: одни и те же UMAP-координаты, но раскрашенные по трём уровням — видно иерархическую детализацию."))

    # X8: Individual metric bars + radar
    cells.append(code("""# Grid of individual metric bars + radar chart
from math import pi

metrics_to_compare = {
    "Silhouette\\n(higher better)": lambda lvl: QUALITY[lvl]["silhouette"],
    "Davies-Bouldin\\n(lower better)": lambda lvl: QUALITY[lvl]["davies_bouldin"],
    "Calinski-Harabasz\\n(higher better)": lambda lvl: QUALITY[lvl]["calinski_harabasz"],
    "Dunn index\\n(higher better)": lambda lvl: QUALITY[lvl]["dunn"],
    "1 − CV\\n(higher better)": lambda lvl: 1 - H.distribution_summary(
        np.bincount(LEVELS[lvl]["labels"][LEVELS[lvl]["labels"] >= 0]))["cv"],
    "1 − Gini\\n(higher better)": lambda lvl: 1 - H.distribution_summary(
        np.bincount(LEVELS[lvl]["labels"][LEVELS[lvl]["labels"] >= 0]))["gini"],
    "Entropy (norm)\\n(higher better)": lambda lvl: H.distribution_summary(
        np.bincount(LEVELS[lvl]["labels"][LEVELS[lvl]["labels"] >= 0]))["entropy_norm"],
}

level_names = ["large", "medium", "small"]
data = {m: [fn(lvl) for lvl in level_names] for m, fn in metrics_to_compare.items()}
colors_bar = ["#4C72B0", "#DD8452", "#55A467"]

# Create grid: 7 individual bar plots + 1 radar
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 4, hspace=0.7, wspace=0.35)

# Individual metric bars (one subplot each)
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
for (r, c), (metric_name, vals_fn) in zip(positions, metrics_to_compare.items()):
    ax = fig.add_subplot(gs[r, c])
    vals = [vals_fn(lvl) for lvl in level_names]
    bars = ax.bar(level_names, vals, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.set_title(metric_name, fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

# Radar chart (right side, spans 2x2)
ax_radar = fig.add_subplot(gs[1:3, 2:], projection="polar")

def normalize_for_radar(values, higher_better=True):
    v = np.array(values, dtype=float)
    if v.max() - v.min() < 1e-9:
        return np.ones_like(v) * 0.5
    normed = (v - v.min()) / (v.max() - v.min())
    if not higher_better:
        normed = 1 - normed
    return normed

# For radar, 'lower better' DB needs inversion
radar_metrics = {
    "Silhouette": (lambda lvl: QUALITY[lvl]["silhouette"], True),
    "Davies-B.": (lambda lvl: QUALITY[lvl]["davies_bouldin"], False),
    "Calinski-H.": (lambda lvl: QUALITY[lvl]["calinski_harabasz"], True),
    "Dunn": (lambda lvl: QUALITY[lvl]["dunn"], True),
    "1−CV": (lambda lvl: 1 - H.distribution_summary(
        np.bincount(LEVELS[lvl]["labels"][LEVELS[lvl]["labels"] >= 0]))["cv"], True),
    "1−Gini": (lambda lvl: 1 - H.distribution_summary(
        np.bincount(LEVELS[lvl]["labels"][LEVELS[lvl]["labels"] >= 0]))["gini"], True),
    "H norm": (lambda lvl: H.distribution_summary(
        np.bincount(LEVELS[lvl]["labels"][LEVELS[lvl]["labels"] >= 0]))["entropy_norm"], True),
}

labels_radar = list(radar_metrics.keys())
N = len(labels_radar)
angles = [n / N * 2 * pi for n in range(N)]
angles += angles[:1]

normed_vals = {}
for mname, (fn, higher) in radar_metrics.items():
    raw = [fn(lvl) for lvl in level_names]
    normed_vals[mname] = normalize_for_radar(raw, higher_better=higher)

for i, lvl in enumerate(level_names):
    values = [normed_vals[m][i] for m in labels_radar]
    values += values[:1]
    ax_radar.plot(angles, values, "o-", linewidth=2.5, label=lvl, color=colors_bar[i])
    ax_radar.fill(angles, values, alpha=0.15, color=colors_bar[i])

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(labels_radar, fontsize=11)
ax_radar.set_ylim(0, 1.05)
ax_radar.set_title("Normalized comparison\\n(higher = better for this level)", fontsize=12, y=1.10)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
ax_radar.grid(True, alpha=0.4)

plt.suptitle("Cross-level metrics comparison", fontsize=16, y=0.995)
plt.savefig(FIG_DIR / "cross_08_metrics_comparison.png", bbox_inches="tight")
plt.show()
"""))
    cells.append(md("> *Plot X8*: многомерное сравнение метрик качества между уровнями."))


def main():
    nb = build()
    # Initialize QUALITY dict in a separate cell (inserted early in notebook)
    # We need QUALITY to exist before level sections use it
    init_cell = new_code_cell("QUALITY = {}")
    # Insert after UMAP cell (position ~10 cells in)
    # Find the position right before "## 1. Large"
    insert_pos = None
    for i, c in enumerate(nb["cells"]):
        if c.cell_type == "markdown" and "## 1." in c.source:
            insert_pos = i
            break
    if insert_pos is not None:
        nb["cells"].insert(insert_pos, init_cell)

    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"Notebook written: {OUT}")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
