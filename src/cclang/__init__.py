"""
CreateCompleteLang (cclang) — interpretable semantic-graph framework.

Public entrypoints are intentionally small; use subpackages:
- cclang.io      : typed I/O (corpora, artifacts, schemas)
- cclang.text    : normalization and tokenization
- cclang.parser  : corpus parser
- cclang.stats   : co-occurrence matrix, weighting
- cclang.lsa     : SVD/LSA on sparse matrices
- cclang.graph   : semantic graph build utilities
- cclang.metrics : pairwise word similarity utilities
- cclang.gaps    : (stubs) gap detectors
- cclang.align   : (stubs) cross-lingual alignment
- cclang.iso     : (stubs) subgraph isomorphism helpers
- cclang.morph   : (stubs) intra-lingual morphology candidates
- cclang.eval    : (stubs) evaluation/reporting
"""
__all__ = ["io", "text", "parser"]
