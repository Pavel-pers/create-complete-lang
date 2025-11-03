# 🌳 Create Complete Language (Создание полного языка)

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
