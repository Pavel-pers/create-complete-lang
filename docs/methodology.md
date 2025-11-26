# Methodology

This project (**CreateCompleteLang**) aims to **detect** and **fill** semantic gaps using an interpretable, graph-based semantic model.

## Pipeline (high level)

1. **Corpus → Normalization → Tokenization**
    - Minimal, language-agnostic normalization (HTML strip, NFKC, optional lower).
    - Tokenizer stubs (Marathi uses a placeholder for now).

2. **Co-occurrence matrix**
    - Symmetric word–word matrix (sliding window).
    - Weighting: TF or **PPMI**.

3. **Interpretable vectors (LSA)**
    - Truncated **SVD** on sparse matrix.
    - Choose `k` via cumulative variance (energy).

4. **Semantic graph**
    - Build **kNN graph** over word vectors (cosine similarity).
    - Export `.graphml` for external tools.

5. **Gap discovery (stubs)**
    - TODO

6. **Cross-lingual & intra-lingual filling (stubs)**
    - TODO
   
7. **Evaluation & report (stubs)**
    - TODO


## Design choices

- **Interpretable LSA/SVD** rather than opaque embeddings.
- **Typed I/O with Pydantic** to keep formats stable and validated.
- **Separation of concerns**: core library vs. thin CLI pipelines.
- **Artifacts** are versioned by schemas, not by ad-hoc adoptions.

## Next steps

- Plug real tokenizers and  language-specific normalization.
- Implement gap detectors and alignment.
- Add ANN search for large vocabularies (FAISS/ScaNN).
