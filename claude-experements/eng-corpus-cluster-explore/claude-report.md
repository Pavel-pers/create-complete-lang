# Отчёт: Cluster Exploration — визуализация и статистика финальных кластеров

## Обзор задачи

Построить Jupyter notebook с визуализациями и статистиками трёх финальных уровней кластеризации английского корпуса, полученных в задаче `eng-corpus-wishart`:

| Уровень | Артефакт | Метод | Кластеры |
|---------|----------|-------|----------|
| Large | `c11-large` | Bisecting K-Means | 18 |
| Medium | `c11-medium` | Bisecting K-Means | 100 |
| Small | `c17-small` | Bisecting K-Means | 517 |

**Эмбеддинги**: `v4-nopc1` — 79,485 × 300, L2-normalized SVD-LSA с удалённым первым главным компонентом.

## Архитектура

```
eng-corpus-cluster-explore/
├── cluster_explore.ipynb     # основной notebook (138 ячеек, ~13 MB)
├── helpers.py                 # helper-функции
├── build_notebook.py          # генератор notebook через nbformat
├── claude-report.md           # этот файл
├── versions/iterations.md     # лог итераций
├── cache/                     # кешированные UMAP/silhouette
│   ├── umap_coords.npy        # (79485, 2)
│   ├── centroids_{large,medium,small}.npy
│   └── silhouette_{large,medium,small}.npz
└── figures/                   # 44 PNG + 3 HTML (plotly)
    ├── {large,medium,small}_01..12_*.png   # 36 per-level plots
    └── cross_01..08_*.{png,html}            # 8 cross-level plots
```

## Метрики качества кластеризации

### Сводная таблица

| Метрика | Large | Medium | Small |
|---------|-------|--------|-------|
| **n_clusters** | 18 | 100 | 517 |
| min / max / mean size | 1119 / 7839 / 4416 | 223 / 1243 / 795 | 95 / 347 / 154 |
| **CV** | 0.491 | 0.324 | 0.307 |
| **max/min ratio** | 7.01 | 5.57 | 3.65 |
| **Gini** | 0.280 | 0.186 | 0.164 |
| **Entropy (norm)** | 0.955 | 0.988 | 0.993 |
| **Silhouette** (cosine, stratified) | 0.054 | −0.006 | −0.065 |
| **Davies-Bouldin** (lower=better) | 7.41 | 7.14 | 6.78 |
| **Calinski-Harabasz** (higher=better) | 544.9 | 173.6 | 45.9 |
| **Dunn index** (centroid-approx) | 0.126 | 0.0014 | 0.0007 |

### Иерархия (purity)

| Связь | Результат |
|-------|-----------|
| Medium → Large | **100/100** кластеров имеют purity > 0.95 |
| Small → Medium | **484/517** кластеров имеют purity > 0.90 (mean purity = 0.978) |

**Интерпретация**: кластеризация формирует **почти идеальную иерархию**. Medium полностью вложен в Large, Small — почти полностью в Medium. Это означает что Bisecting K-Means сохраняет hierarchy-structure.

## Ключевые наблюдения

### Размеры и распределение

1. **Равномерность распределения улучшается с гранулярностью**: CV 0.49 → 0.32 → 0.31, Gini 0.28 → 0.19 → 0.16. На Small-уровне распределение почти идеально (entropy_norm = 0.993).
2. **max/min ratio уменьшается**: 7.0 → 5.6 → 3.7 — при большем числе кластеров размеры выравниваются.
3. **Rank-size log-log** показывает мягкое убывание без чётного Zipf-закона — характерно для продвинутой кластеризации.

### Качество (silhouette)

- **Silhouette падает с гранулярностью** (0.054 → −0.006 → −0.065). Это ожидаемо:
  - На Large кластеры тематически широкие → точки сильнее тяготеют к своему центроиду
  - На Small темы сужаются и часто пересекаются в embedding-пространстве → многие точки ближе к соседнему кластеру
  - Отрицательные значения на Medium/Small не означают "плохую" кластеризацию — это следствие **семантической близости смежных тем** (кулинария vs. напитки, одежда vs. ткани и т.п.)
- **Davies-Bouldin** (7.4 → 7.1 → 6.8) даже немного улучшается при большем k — кластеры становятся более компактными относительно своего разброса.
- **Calinski-Harabasz** падает (545 → 174 → 46) — ожидаемо, так как между-кластерная дисперсия делится на больше кластеров.

### Пространственная структура (UMAP)

- **UMAP-триптих** (`cross_07_umap_triptych.png`): одни и те же 2D координаты, раскрашенные по трём уровням. Large-кластеры образуют крупные цветовые массивы; на Medium они разбиваются на подобласти; на Small — на мелкие островки внутри подобластей. Визуально подтверждается иерархия.
- **Центроидная геометрия** (`{level}_08_centroid_heatmap.png`): на Large видны чёткие блоки (архаические стили vs. современные vs. поэтические). На Small (517×517) виден диагональный блочный паттерн — соседние по dendrogram-ordering кластеры близки, что подтверждает иерархическую согласованность.

### Дендрограммы (`{level}_09_dendrogram.png`)

- **Large**: 18 кластеров образуют 3-4 верхне-уровневых "мега-группы" ≈ стилистические эпохи (современная / викторианская / архаичная поэзия / дикий запад).
- **Medium**: 100 кластеров дробятся на ~10-15 под-семейств — на той же высоте dendrogram что и Large. Визуально подтверждает совместимость уровней.
- **Small**: truncated на 50 leaves — иерархия глубокая, но хорошо структурированная.

### Межкластерные связи

- **Flow Large→Medium** (`cross_01_flow_LM.png`): **идеальная диагональная матрица** (после сортировки Medium по доминирующему Large). Каждый Medium полностью принадлежит ровно одному Large.
- **Flow Medium→Small** (`cross_02_flow_MS.png`): более размытая диагональ, но покрытие >90% — Small иногда разделяется между двумя соседними Medium на стыках тем.
- **Sankey/Treemap/Sunburst** (`cross_03/04/05.html`): интерактивные визуализации иерархии, цвет наследуется от Large-прародителя.

## Артефакты

### Локальные файлы

```
figures/
├── large_01_sizes_sorted.png         # распределение размеров
├── large_02_rank_size_loglog.png     # rank-size закон
├── large_03_density_kde.png          # per-cluster density distribution
├── large_04_intra_dist_boxplot.png   # компактность
├── large_05_silhouette_plot.png      # классический silhouette
├── large_06_silhouette_per_cluster.png
├── large_07_size_vs_compactness.png
├── large_08_centroid_heatmap.png     # межкластерные расстояния
├── large_09_dendrogram.png           # иерархия центроидов
├── large_10_network.png              # графy близости
├── large_11_umap_clusters.png        # 2D проекция
├── large_12_umap_density.png         # density overlay
├── medium_01..12_*.png               # аналогично для Medium
├── small_01..12_*.png                # аналогично для Small
├── cross_01_flow_LM.png              # Large→Medium heatmap
├── cross_02_flow_MS.png              # Medium→Small heatmap
├── cross_03_sankey.{png,html}        # Sankey diagram
├── cross_04_treemap.{png,html}       # Treemap
├── cross_05_sunburst.{png,html}      # Sunburst
├── cross_06_stacked_bar.png          # Large stacked by Medium
├── cross_07_umap_triptych.png        # 1×3 UMAP by level
└── cross_08_metrics_comparison.png   # multi-metric + radar
```

### Методология

- **UMAP**: один раз на 79,485 × 300 embeddings (cosine metric, n_neighbors=30, min_dist=0.1, random_state=42), кеш `cache/umap_coords.npy`. Центроиды проецируются через `reducer.transform()`.
- **Silhouette**: стратифицированная выборка с гарантией ≥20 точек на кластер. Sample sizes: Large=20k, Medium=10k, Small=15k. Cosine metric.
- **Davies-Bouldin, Calinski-Harabasz**: на полных данных (O(n·k), дёшево).
- **Dunn index**: centroid-based approximation = `min(inter-centroid dist) / max(2*radius)`. Точный Dunn требует полных попарных расстояний — невыполнимо для 517 кластеров.
- **Flow matrices**: `np.add.at` на парных меток.
- **Top-20 + Random-20 selection**: детерминированно (seed=42), используется только в Small per-cluster визуализациях.

### Оговорки

1. **Roles** (поле из `labels.npz`) все нули для Bisecting K-Means — не используется.
2. **Densities** (также из `labels.npz`) — это расстояния до центроида внутри кластера, отражают внутреннюю структуру K-Means и показываются в KDE (plot 03) и UMAP-оверлее (plot 12).
3. **Silhouette negative ≠ bad clustering** на словарных эмбеддингах: семантические кластеры перекрываются по границам, но лингвистически остаются разделимыми (см. `claude-clustering-review.md` предыдущей задачи).

## Воспроизведение

```bash
# Установить зависимости
pip install seaborn umap-learn plotly squarify jupyter kaleido

# Сгенерировать notebook
cd claude-experements/eng-corpus-cluster-explore
python build_notebook.py

# Выполнить
jupyter nbconvert --to notebook --execute --inplace cluster_explore.ipynb \
    --ExecutePreprocessor.timeout=3600
```

После первого запуска UMAP кешируется → повторные прогоны <30 сек.

## Итерации

См. `versions/iterations.md`. Основные фиксы:
- **v1 → v2**: исправлены UMAP centroid annotations (white text на чёрном круге теперь viewable через path_effects), переверстаны flow_LM heatmap (transpose для лучшего aspect ratio), X8 metrics comparison расширен до grid с индивидуальными bar plots + radar.
