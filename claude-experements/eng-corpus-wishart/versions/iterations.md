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

## Phase 2 — Дополнительные эксперименты (dim=50, CBOW, CBOW→SVD)

### План

Цели:
1. Проверить, можно ли достичь сопоставимого (или лучшего) качества кластеризации с эмбеддингами **~50 измерений** (вместо 300)
2. Попробовать **CBOW (Word2Vec)** как альтернативу SVD-LSA
3. Попробовать гибрид **CBOW → SVD** — обучить CBOW в высокой размерности, затем снизить через SVD

Все эксперименты на том же словаре (min_tf=5, min_df=3) → 79,485 слов.

### Версии эмбеддингов

| Version | Метод | Size | Постобработка |
|---------|-------|------|---------------|
| v4-nopc1 | SVD-LSA log-entropy k=300 | 300 | center + PC1 removal + L2 | **baseline** |
| **v7-svd50** | SVD-LSA log-entropy k=50 | 50 | center + L2 |
| **v7-svd50-nopc1** | SVD-LSA log-entropy k=50 | 50 | center + PC1 removal + L2 |
| **v8-cbow50** | Word2Vec CBOW vector_size=50 | 50 | center + L2 |
| **v8-cbow50-nopc1** | Word2Vec CBOW vector_size=50 | 50 | center + PC1 removal + L2 |
| **v9-cbow300** | Word2Vec CBOW vector_size=300 | 300 | center + L2 |
| **v9-cbow300-svd50** | v9-cbow300 → SVD → 50 | 50 | center + L2 |
| **v9-cbow300-svd50-nopc1** | v9-cbow300 → SVD → 50 | 50 | center + PC1 removal + L2 |

### Стратегия

1. Построить все версии эмбеддингов
2. Прогнать Bisecting K-Means 3-уровневую кластеризацию (target=18/100/600) на каждой
3. Сравнить метрики: CV, max/min, silhouette, DB, CH, Dunn
4. Для топ-кандидатов — лингвистическая валидация
5. Обновить отчёт с выводами

## Phase 2 — Результаты

### Построенные эмбеддинги
- **v7-svd50**: SVD-LSA log-entropy k=50 (sigma_power=0.5, center+L2)
- **v7-svd50-nopc1**: + удаление PC1
- **v8-cbow50**: Word2Vec CBOW vector_size=50, window=5, epochs=10
- **v8-cbow50-nopc1**: + удаление PC1
- **v9-cbow300**: Word2Vec CBOW vector_size=300
- **v9-cbow300-svd50**: v9-cbow300 → randomized_svd k=50 (center + L2)
- **v9-cbow300-svd50-nopc1**: + удаление PC1

### Сравнение метрик Bisecting K-Means на всех версиях

| Version | Dim | L CV | L ratio | M CV | M ratio | S CV | S ratio |
|---------|-----|------|---------|------|---------|------|---------|
| v4-nopc1 (baseline Phase 1) | 300 | 0.491 | 7.01 | 0.324 | 5.57 | 0.307 | 3.65 |
| v7-svd50 | 50 | 0.428 | 3.83 | 0.399 | 6.03 | 0.348 | 4.31 |
| v7-svd50-nopc1 | 50 | 0.356 | 5.18 | 0.252 | 4.51 | 0.266 | 3.48 |
| v8-cbow50 | 50 | 0.265 | 2.48 | **0.225** | **2.62** | 0.267 | 3.92 |
| **v8-cbow50-nopc1** ★ | 50 | 0.271 | 2.53 | 0.226 | 2.93 | **0.240** | **3.06** |
| v9-cbow300 | 300 | 0.260 | 2.77 | 0.248 | 9.15 | 0.284 | 3.61 |
| v9-cbow300-svd50 | 50 | **0.191** | **1.99** | 0.271 | 7.47 | 0.262 | 3.04 |
| v9-cbow300-svd50-nopc1 | 50 | 0.233 | 2.28 | 0.240 | 4.07 | 0.273 | 3.63 |

### Ключевые выводы Phase 2

1. **CBOW существенно превосходит SVD-LSA** по равномерности кластеров — CV упал с 0.3-0.5 до 0.22-0.27.
2. **50-мерные эмбеддинги работают лучше 300-мерных** для задачи кластеризации (меньше шума, лучшая геометрия).
3. **PC1 removal критичен для SVD-LSA**, но **менее важен для CBOW** — CBOW естественно распределяет дисперсию более равномерно (σ₁/σ₂ ≈ 1.3 против ≈2.4 у SVD-LSA).
4. **CBOW→SVD гибрид** даёт лучшую формальную равномерность (CV=0.191, ratio=1.99 на Large), но **ухудшает семантику** — кластеры становятся тематически размытыми (смесь диалектов, имён, чисел).
5. **v8-cbow50-nopc1** — лучшее решение по совокупности: отличная равномерность И лучшая лингвистическая связность.

### Лингвистическая валидация v8-cbow50-nopc1

**Large (18 кластеров)**:
- C8: материалы/фактура (silvered, crinkled, ribbed, shiny)
- C4: география (sandhill, pyrenee, bay, falls, foreshore)
- C12: психология (defensiveness, passivity, unawareness)
- C13: западный диалект (sassy, danged, doggone, pardner)
- C10: персонажи романов (marchmont, newman, vanbrugh)
- C9: ботаника/наука (spheroidal, laminar, hirsuta, nucleolus)
- C2: несправедливость (unjust, misrepresentation, forgive)
- C7: шотландский (naither, yoong, mayna, neebour)
- C3: еда (doughnut, sardine, pork, crock)
- C11: юриспруденция (trustee, solicitor, authorize)
- C0: средневерхне-архаика (messire, hath, erst, thee)
- C15: елизаветинский (thinke, onely, keepe, euill)
- C17: феодальная знать (baron, emperor, valentinian)
- C14: французский (ceux, demande, quant, heureux)
- C16: военные топонимы (ptolemais, carnarvon, stockaded)

Все 18 кластеров тематически связны, явных "свалок" нет.

**Small (538 кластеров)** — примеры:
- Arthurian romance (yolande, lancelot, beauteous, tressed, goddess)
- Деревья (acacia, rhododendron, mimosa, ilex, sycamore)
- Химия (nitrite, amyl, hydrochloric, alcohol)
- Политика США (nominate, legislature, congress, elect)
- Удивление (awestruck, transfixed, wonderingly, fascinated)
- Еда/завтрак (oatmeal, tapioca, marmalade, cake)
- Французский, шотландский — отдельные тугие кластеры

### Рекомендованная финальная версия

**v8-cbow50-nopc1** с Bisecting K-Means:
- `c_p2_v8pc1-large` — 18 кластеров, CV=0.271
- `c_p2_v8pc1-medium` — 100 кластеров, CV=0.226
- `c_p2_v8pc1-small` — 538 кластеров, CV=0.240

Превосходит Phase 1 baseline по всем метрикам **И** по лингвистической валидации.

## Final best results (Phase 1)

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
