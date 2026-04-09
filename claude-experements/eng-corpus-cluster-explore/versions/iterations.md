# Iteration Log — Cluster Exploration Notebook

## v1 — Initial generation
- Built `helpers.py` module + `cluster_explore.ipynb` via `build_notebook.py`
- 138 cells, full execution successful (no errors)
- Generated 44 PNG figures, 3 HTML files (Sankey/Treemap/Sunburst)
- UMAP fit time: ~3 min (cached)

## v1 — Visual issues found
After reading exported PNGs:

1. **L11 (Large UMAP)**: centroid numbers inside black circles not legible (text rendering issue)
2. **M11 (Medium UMAP)**: yellow annotation boxes cover cluster colors
3. **S11 (Small UMAP)**: too many yellow boxes, cluster colors invisible
4. **X7 (UMAP triptych)**: титулы слишком мелкие
5. **X1 (flow_LM heatmap)**: соотношение сторон 100×18 — картинка слишком узкая
6. **S8 (small centroid heatmap)**: нет title/xlabel/ylabel (хотя cbar есть)
7. **X8 (metrics comparison bars)**: CH-log доминирует по высоте, остальные метрики неразличимы

## v2 — Fixes
- UMAP centroid annotations: убрать bbox, использовать круглые маркеры с цифрами внутри через plt.text с bold font
- Small/Medium UMAP: только 10 крупнейших подписать, остальные просто маркеры без подписи
- X7: suptitle fontsize=16, subplot titles fontsize=13
- X1: figsize=(16,10), transpose (Large по X, Medium по Y) — 100 меток на Y читается
- S8: явные title/labels
- X8: раздельные subplots для каждой метрики (5 bar charts + radar) вместо одного bar + radar
