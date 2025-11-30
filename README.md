# Create Complete Language

**The Tree of Knowledge** is a research project that detects and fills **semantic gaps**—missing or unevenly distributed concepts—in human languages using **semantic graphs** built from interpretable vector spaces (e.g., truncated **SVD/LSA**).  
Primary focus language: **Marathi (मराठी)**, with cross-lingual comparison to a base language.

---

## Overview

- **Nodes** = words; **edges** = semantic proximity (cosine similarity in a low-dimensional space from SVD).
- **Goal**: reveal conceptual asymmetries and propose data-backed ways to bridge them.
- **Two tracks**
  1. **Cross-lingual**: find near-isomorphic subgraphs across languages (e.g., Marathi ↔ base language) to surface missing nodes/links.  
  2. **Intra-lingual**: detect and complete patterns inside a single language via analogies and graph motifs.

---

## Why interpretable?

We prefer **interpretable** factor models (SVD/LSA) over opaque embeddings to trace each semantic relation back to corpus statistics and factors.

---

## Methodology (high level)

1. **Corpora & Preprocessing**: clean, tokenize (language-specific), optional lemmatization.  
2. **Co-occurrence Matrix**: word × context (windowed), weighted by PMI/TF-IDF.  
3. **Truncated SVD**: derive k-dimensional word vectors.  
4. **Graph Construction**: k-NN or thresholded cosine → weighted edges.  
5. **Gap Discovery**: cluster analysis, link prediction heuristics, motif search, stable multi-word periphrases.  
6. **Gap Filling**:  
   - _Cross-lingual_: align spaces with a small seed dictionary (linear/orthogonal mapping), search isomorphic subgraphs, import missing node candidates.  
   - _Intra-lingual_: analogy patterns, paradigm completion, morphology-based candidates.  
7. **Human Validation**: expert/annotator review loop.

---

## Functional map

- Core utilities: logging (`cclang.common.logx`), S3 config (`cclang.config.s3`), file manager (`cclang.ingest.fs`), S3 store (`cclang.io.cloud`), manifests (`cclang.io.manifest`), fetched-items registry (`cclang.io.fetched_items_store`), network ingest (`cclang.ingest.net`), task queue logging (`cclang.models.tasks_queue`).
- Pipelines: discover links (`pipelines.corpus.discover_links` → `cclang-discover-links`), fetch PDFs (`pipelines.corpus.fetch_pdfs` → `cclang-fetch-pdfs`).
- Utility script: upload local data to S3 (`scripts.upload_local_to_s3` → `upload-local-to-s3`).

See `docs/functional_overview.md` for a concise guide to configuration, data layout, and CLI examples.
