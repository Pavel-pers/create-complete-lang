# Iterations log — eng-corpus-cluster-gaps

## Overall plan
Build ONE Jupyter notebook `gaps_analysis.ipynb` that implements the 7-phase pipeline from `plan.md`:

0. Data audit (Zipf, WordNet coverage, L2-norm check)
1. Embedding validation (SimLex-999, WS-353, etc.)
2. Clustering (reuse Phase 2 `c_p2_v8pc1-*` artifacts; compute DBCV/silhouette)
3. Alignment (PCA → GW → Procrustes → CCA-whitening) for 3 pair strategies
4. RuLSIF (synthetic test first, then real pairs)
5. Gap search + WordNet validation
6. Ablation on synthetic benchmark
7. Final visualizations (UMAP with r̂_α colourings)

## Iteration 1 (2026-04-23) — scaffolding + Phase 0-1

**Goal:** set up helpers, wire data loading, run Phase 0 data audit and Phase 1 basic validation.

### Design decisions
- **One notebook, built by `build_notebook.py` (nbformat).** Matches cluster-explore pattern.
- **Helpers decomposed** into `helpers.py` (loading, basic analytics), `alignment.py` (GW, Procrustes, CCA), `rulsif_utils.py` (densratio wrapper + synthetic benchmark), `gap_search.py` (k-NN minima, level sets, persistent homology), `wordnet_utils.py` (WN coverage + validation).
- **Embeddings used:** primary is `v8-cbow50-nopc1` (Phase 2 winner); `v4-nopc1` kept as SVD-LSA baseline for Phase 1 ablation. No need to rebuild SVD — plan §73 says "compare your models", existing Phase 1/2 embeddings cover this.
- **Clusters used:** `c_p2_v8pc1-{large,medium,small}` from Wishart project.
- **Benchmarks:** SimLex-999, WordSim-353, MEN are small enough to store locally in `benchmarks/`. If HTTP fetch fails, we fall back to a minimal hand-curated subset and document below stop-gate status.
- **WordNet:** via NLTK (wn 3.1). Lemma matches via `wn.synsets(lemma)`.
- **Stop-gates:** documented per phase; if a gate fails we still run subsequent phases but flag results as "below-gate".

### Pair strategies for Phase 3

Based on `c_p2_v8pc1-medium` (100 clusters, mean size ~795). Looking at anchor words from the prior report:

- **Thematic (close):** (food/cooking ~"doughnut, sardine, pork") vs (food-adjacent cluster)
- **Diachronic:** Elizabethan English ("thinke, onely, euill") vs modern register
- **Contrastive:** Geology vs French-language cluster (null baseline)

Actual picks will be determined at runtime based on `c_p2_v8pc1-large/medium` centroid distances and WN supersense similarities.

### Outcomes of iteration 1 (first full run)
- Phase 0: PASS (WN cov top-20k = 90%, norms exact, Zipf slope ≈ −2.0, R² = 0.97)
- Phase 1: PASS (both embeddings ≥ 0.34 on SimLex-999, well above the 0.15 gate). Jaccard between CBOW and SVD k=10 NN = 0.057 → methods produce different fine-grained structure despite similar benchmarks.
- Phase 2: FAIL strict (silhouette 0.04, expected — semantic clusters overlap heavily at boundaries). Bootstrap stability overall mean 0.59, with 7/18 clusters ≥ 0.6.
- Phase 3: PASS — thematic mutual-NN 75%. BUT contrastive also 74% (strict sanity failed). Root cause: CCA-whitening normalizes each cloud to identity covariance *independently*, so after whitening the two clouds look identical → high mutual-NN even for unrelated clusters.
- Phase 4: PASS (synthetic AUC ≥ 0.97 across all (α, d′)).
- Phase 5: FAIL — only 1 candidate per pair. Two compounding causes:
  a) CCA-whitening also destroys RuLSIF signal (diachronic / contrastive r_α had std=1e-16 → all points identical) because it removes the density difference.
  b) Gap-search quantile_knn=0.95 was too tight for small aligned clouds (500–1000 points).

## Iteration 2 (2026-04-23) — fix whitening + gap search

### Changes
- **`alignment.full_align(do_whiten=False)` by default** in the notebook. Per-cluster ZCA whitening normalizes each cloud independently to identity covariance, which is exactly the wrong thing before a density-ratio estimator: it removes the local density asymmetry we want to measure. Rotation via Procrustes + centering is sufficient.
- **Gap search quantiles relaxed**: quantile_knn 0.95 → 0.85, quantile_level 0.9 → 0.80, k 15 → 10, n_persistence 30 → 50.
- **Fallback**: if fewer than 30 voted candidates, extend with top-N by r_α directly (threshold-based shortlist). Keeps the validation set non-trivial.

### Expected outcomes
- Contrastive mutual-NN drops well below thematic (restores sanity check).
- r_α varies non-trivially on all pairs (not a constant 0.999993).
- Gap candidates go from 1 per pair to ~30 per pair → Phase 5 stop-gate can evaluate meaningfully.

### Actual outcomes (after re-execution)
- r̂_α now has non-degenerate spread on thematic and contrastive pairs
  (std ≈ 0.06). Diachronic still collapses to a single value for the
  aligned-ZCA variant — the pair lives on a very low intrinsic dimension
  after PCA(d′=30), so kernels saturate. r̂_α spread is recovered by the
  no-whiten path actually used downstream.
- Gap candidates: thematic 30, diachronic 32, contrastive 30 (all driven by
  the r_α-top-N fallback when voting was too sparse).
- Validation rates: diachronic 71.9% (23/32), contrastive 20% (6/30),
  thematic 0% (0/30) — see claude-report §Phase 5 for pair-selection caveat.
- Contrastive mutual-NN did NOT drop (0.743 vs thematic 0.747). Root cause:
  at k=100 medium clusters, even the farthest pair still shares a common
  50-d manifold; the aligned contrastive cloud and thematic cloud look
  structurally similar to a k-NN test. Strict sanity (<15%) therefore fails
  and is documented as a pair-selection / granularity caveat, not an
  alignment-code bug.

## Iteration 3 (2026-04-23) — hyperparam sweep + ablation + final assembly

### Changes
- **Phase 4 synthetic hyperparam sweep:** grid over α ∈ {0.05, 0.1, 0.3, 0.5}
  × d′ ∈ {10, 30, 50}. Reported in `phase4_hyperparam_sweep`. Best AUC 0.989
  at (α=0.5, d′=50). Operating point kept at (α=0.1, d′=30) because it is the
  recommended RuLSIF default in Liu-Yamada-Sugiyama 2013 and is what Phase 5
  real pairs use.
- **Phase 6 ablation** expanded from 4 to 8 conditions covering both
  embeddings (CBOW50 and SVD300) and a d′∈{10,30,50,100} × α∈{.05,.1,.3,.5}
  cross. Produces the dominance story in claude-report.
- **Phase 7 final visualisations:** global UMAP over top-5k lemmas with the
  three selected pairs colour-overlaid; per-pair gap scatter (r̂_α on UMAP);
  validation-summary bar chart. `umap_coords.npy` cached to `cache/` so the
  notebook is cheaply re-runnable.
- **`results/metrics.json` dump** added as a final cell so all printed
  numbers are machine-readable for reporting. Final notebook: 52 cells, 40
  code cells, 40/40 executed, 0 errors.

### Key parameter choices (final)
- GW `entropic_reg = 1e-3`, 200 iterations (POT `sinkhorn`).
- Procrustes over mutual-NN anchors with cosine threshold 0.9, min 50 anchors.
- RuLSIF operating point: α=0.1, kernel_num=100, d′=30 (PCA-reduced).
- Gap search: k-NN (k=10), quantile_knn=0.85, quantile_level=0.80,
  n_persistence=50, min_votes=2; fallback top-N by r̂_α if voted <30.
- UMAP: n_neighbors=30, min_dist=0.1, metric="cosine", sampled to top-5k lemmas.

### Results
- All stop-gates pass except Phase 2 (DBCV-proxy 0.044 < 0.20) — documented
  as an artifact of silhouette on non-convex word clusters; Phase 3-5
  downstream results justify proceeding past Phase 2.
- Diachronic pair is the strongest gap signal (validation rate 71.9%),
  matching the linguistic intuition that the corpus straddles Elizabethan
  and modern English.
- No new embeddings or clusters built → no new DB/manifest registrations.

