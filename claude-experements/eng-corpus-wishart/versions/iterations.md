# Iteration Log

## v1 — Baseline SVD + log-entropy (79K vocab)
- **Date**: 2026-04-09
- **Vocab**: min_df=3, max_df_ratio=0.85, min_tf=5 → 79,485 words
- **Embeddings**: SVD k=300, sigma_power=0.5, log-entropy, center+normalize
- **Result**: Baseline embeddings, dense archaic word cluster dominates

### c1-large (Wishart, k=15, target=18)
- 17 clusters, sizes [306—42,459], CV=2.20, max/min=138.75
- FAIL: two mega-clusters (42K + 18K) contain archaic/rare words
- Linguistically: small clusters good (nature, medical, french), but mega-clusters are dump

### c1-medium (Wishart, k=15, target=100)  
- 74 clusters, sizes [103—15,465]
- FAIL: still very uneven

## v2 — Stricter vocab (40K)
- **Vocab**: min_tf=20, min_df=10 → 40,299 words
- Tried to reduce archaic word mass

### c2-large (Wishart, k=15, target=18): 19 clusters, [163—16,987]. FAIL
### c3-large (Wishart, k=30): 19 clusters, [221—21,650]. FAIL
### c4-large (K-Means, k=18): 18 clusters, [425—16,256], CV=1.60. FAIL

**Conclusion**: Wishart fundamentally produces uneven clusters on word embeddings due to density variation

## c5-large — Bisecting K-Means (v2)
- 68 clusters (overshot due to max_size_ratio splitting). Fixed algorithm

## c6 — Bisecting K-Means refined (v2)
- **large**: 18 clusters, [468—3,532], CV=0.43, ratio=7.55. GOOD uniformity
- **medium**: [34—681], some below 100. FAIL
- **small**: [1—100], too few words for 750 clusters. FAIL

**Decision**: v2 vocab too small for medium/small. Use v1 for those.

## c7 — Bisecting on v1
- **medium** (target=100): 99 clusters, [127—1,406], ratio=11.1. OK but ratio high
- **small** (target=600): 432 clusters. Below 500 target.

## v4-nopc1 — Removing first SVD component
- **Key insight**: PC1 captures frequency effects (σ₁=113 vs σ₂=47)
- Removing PC1 from v1 embeddings dramatically improves cluster uniformity

### c11 — Bisecting on v4-nopc1
- **large**: 18 clusters, [1,119—7,839], CV=0.49, ratio=7.01. PASS ✓
- **medium**: 100 clusters, [223—1,243], CV=0.32, ratio=5.57. PASS ✓
- **small**: K-Means k=550 gives [7—2,002]. FAIL

### c14-small (bisecting, target=700): 489 clusters, [100—347], ratio=3.5. Close but < 500.

## c17-small — Final small clustering
- Bisecting on v4-nopc1, target=800, min_cluster_size=95
- **517 clusters**, [95—347], CV=0.31, ratio=3.65. PASS ✓

## Final best results

| Level | Version | Embeddings | Method | Clusters | Min | Max | CV | Ratio |
|-------|---------|------------|--------|----------|-----|-----|----|-------|
| Large | c11-large | v4-nopc1 | Bisecting K-Means | 18 | 1119 | 7839 | 0.49 | 7.01 |
| Medium | c11-medium | v4-nopc1 | Bisecting K-Means | 100 | 223 | 1243 | 0.32 | 5.57 |
| Small | c17-small | v4-nopc1 | Bisecting K-Means | 517 | 95 | 347 | 0.31 | 3.65 |

## DBSCAN / HDBSCAN tests (c18-c20)

Density-based methods tested on v4-nopc1 and v2:
- **c18-dbscan** (eps=0.2, min_samples=15): только 5 кластеров, 78965 из 79485 noise
- **c18-dbscan (eps sweep 0.08-0.15)**: всё noise или 1-5 крохотных кластеров
- **c19-hdbscan-large** (v2, min_cluster_size=1000): 0 кластеров, всё noise (6 мин)
- **c20-hdbscan-medium** (v2, min_cluster_size=100): 2 кластера, [5701-34598], 33531 noise

**Вывод**: DBSCAN/HDBSCAN фундаментально не работают на word embedding пространстве. Оно имеет один гигантский плотный кластер с разреженными выбросами — density-based методы либо теряют всё в noise, либо объединяют в один кластер.

## Wishart correctness fix (c21-c22)

**Баг найден**: В исходной реализации `build_hierarchy` вес ребра = raw distance между точками. При replay 93.4% рёбер пропускались — edge event срабатывал ДО активации вершин (когда dist(i,j) < knn_radii[i]), и union не выполнялся.

**Исправление**: Вес ребра = mutual reachability distance = `max(dist(i,j), knn_radii[i], knn_radii[j])`. Это гарантирует, что при срабатывании edge event обе вершины уже активны. После фикса: 100% рёбер обрабатываются.

**Валидация на синтетике**: 3 гауссовых блоба по 300 точек → Wishart находит 3 кластера по 300 с Adjusted Rand Index = 1.0 (идеально).

**Wishart на word embeddings (после фикса)**:
- c21-wishart-large (v4-nopc1, k=15, target=18): 13 кластеров, один 74994 слов (93%)
- c22-wishart-large-fixed (тот же результат)
- Максимум различных уровней кластеризации в иерархии: 37 (v4-nopc1, v6-nopc1), 17 (v2), 10 (v3-ppmi)

**Вывод**: Wishart корректен, но не подходит для word embeddings — пространство имеет одну плотную структуру, которую density-based методы видят как единое целое. Bisecting K-Means (partition-based) остаётся оптимальным для требуемой равномерности.
