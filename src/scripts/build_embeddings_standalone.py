"""
Standalone pipeline: pre-lemmatized corpus → vocab → TDM → SVD → embeddings.

Replicates the cclang embedding pipeline without database or cloud.
Produces artifacts in the same format as the main pipeline.

Input: plain-text file where each line is a document, tokens are space-separated,
       sentences are delimited by 2+ consecutive spaces.

Output structure:
    <output-dir>/
    ├── vocab.tsv                          # idx, lemma, tf, df
    ├── tdm/
    │   ├── matrix.npz                     # sparse CSR (n_fragments × vocab_size)
    │   └── row_index.tsv                  # row_idx, fragment_id
    ├── svd/
    │   ├── U.npy, sigma.npy, Vt.npy
    │   └── stats.json
    └── embeddings/
        ├── word_vectors.npy               # V·Σ^α  (vocab_size × k)
        ├── meta.json
        └── word_vectors_center_norm/
            ├── word_vectors.npy
            └── meta.json

Usage:
    python src/scripts/build_embeddings_standalone.py \\
        --corpus data/artifacts/archive/english_newlit_corpus.txt \\
        --output-dir data/artifacts/english_newlit \\
        -k 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.utils.extmath import randomized_svd


# ── Step 1: build vocabulary ────────────────────────────────────────────────

def build_vocab(
    corpus_path: Path,
    min_df: int,
    max_df_ratio: float,
    min_tf: int,
) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray,
           list[list[list[str]]]]:
    """
    Build vocabulary from pre-lemmatized corpus.

    Returns (idx2word, word2idx, tf, df, documents)
    where documents = list of docs, each doc = list of sentences,
    each sentence = list of tokens.
    """
    print("[vocab] reading corpus …")
    documents: list[list[list[str]]] = []
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences = [
                s.strip().split()
                for s in re.split(r" {2,}", line)
                if s.strip()
            ]
            documents.append(sentences)

            doc_lemmas: set[str] = set()
            for sent in sentences:
                for tok in sent:
                    tf_counter[tok] += 1
                    doc_lemmas.add(tok)
            for lemma in doc_lemmas:
                df_counter[lemma] += 1

    n_docs = len(documents)
    max_df_abs = int(n_docs * max_df_ratio)

    print(f"[vocab] documents: {n_docs}, raw vocab: {len(tf_counter)}")
    print(f"[vocab] filtering: min_tf={min_tf}, min_df={min_df}, "
          f"max_df={max_df_abs} ({max_df_ratio:.0%})")

    filtered = sorted(
        (
            (lemma, tf_counter[lemma], df_counter[lemma])
            for lemma in tf_counter
            if tf_counter[lemma] >= min_tf
            and df_counter[lemma] >= min_df
            and df_counter[lemma] <= max_df_abs
        ),
        key=lambda x: -x[1],
    )

    idx2word = [item[0] for item in filtered]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    tf_arr = np.array([item[1] for item in filtered])
    df_arr = np.array([item[2] for item in filtered])

    print(f"[vocab] filtered vocab: {len(idx2word)}")
    return idx2word, word2idx, tf_arr, df_arr, documents


def save_vocab(
    path: Path,
    idx2word: list[str],
    tf: np.ndarray,
    df: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("idx\tlemma\ttf\tdf\n")
        for i, word in enumerate(idx2word):
            f.write(f"{i}\t{word}\t{tf[i]}\t{df[i]}\n")
    print(f"[vocab] saved → {path}")


# ── Step 2: build fragments ─────────────────────────────────────────────────

def build_fragments(
    documents: list[list[list[str]]],
    word2idx: dict[str, int],
    fragment_size: int,
    min_fragment_ratio: float = 0.5,
) -> tuple[list[list[int]], list[str]]:
    """
    Split documents into fragments of *fragment_size* sentences,
    convert tokens to vocab IDs.

    Returns (fragments, fragment_ids).
    """
    print(f"[fragments] chunking into {fragment_size}-sentence fragments …")
    fragments: list[list[int]] = []
    fragment_ids: list[str] = []
    min_sents = max(1, int(fragment_size * min_fragment_ratio))

    for doc_idx, doc in enumerate(documents):
        for start in range(0, len(doc), fragment_size):
            chunk = doc[start : start + fragment_size]
            if len(chunk) < min_sents:
                continue
            token_ids = [
                word2idx[tok]
                for sent in chunk
                for tok in sent
                if tok in word2idx
            ]
            if not token_ids:
                continue
            frag_hash = hashlib.sha256(
                f"{doc_idx}:{start}:{token_ids}".encode()
            ).hexdigest()[:16]
            fragments.append(token_ids)
            fragment_ids.append(frag_hash)

    print(f"[fragments] total: {len(fragments)}")
    return fragments, fragment_ids


# ── Step 3: build TDM ───────────────────────────────────────────────────────

def log_entropy_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Log-entropy weighting (same as build_tdm.py)."""
    local = tf_matrix.copy()
    local.data = np.log1p(local.data)

    n_docs = tf_matrix.shape[0]
    csc = tf_matrix.tocsc()
    col_sums = np.array(csc.sum(axis=0)).flatten()

    col_indices = np.repeat(np.arange(csc.shape[1]), np.diff(csc.indptr))
    p = csc.data / col_sums[col_indices]

    p_log_p = p * np.log(p)
    entropy = np.zeros(csc.shape[1])
    np.add.at(entropy, col_indices, -p_log_p)

    global_weights = 1.0 - entropy / np.log(n_docs)
    return local.multiply(global_weights)


def tfidf_weighting(tf_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """TF-IDF weighting (same as build_tdm.py)."""
    local = tf_matrix.copy()
    local.data = np.log1p(local.data)

    n_docs = tf_matrix.shape[0]
    csc = tf_matrix.tocsc()
    df = np.diff(csc.indptr)
    idf = np.log(n_docs / df)
    return local.multiply(idf)


WEIGHTING = {"log-entropy": log_entropy_weighting, "tf-idf": tfidf_weighting}


def build_tdm(
    fragments: list[list[int]],
    fragment_ids: list[str],
    vocab_size: int,
    weighting: str,
    out_dir: Path,
) -> sparse.csr_matrix:
    """Build weighted Term-Document Matrix and save artifacts."""
    print(f"[tdm] building sparse matrix ({weighting}) …")
    rows, cols, vals = [], [], []
    for i, frag in enumerate(fragments):
        for tid, cnt in Counter(frag).items():
            rows.append(i)
            cols.append(tid)
            vals.append(cnt)

    tf_matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(fragments), vocab_size),
        dtype=np.float64,
    )
    print(f"[tdm] raw shape: {tf_matrix.shape}, nnz: {tf_matrix.nnz}")

    result = WEIGHTING[weighting](tf_matrix)

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "matrix.npz", result)
    with open(out_dir / "row_index.tsv", "w", encoding="utf-8") as f:
        f.write("row_idx\tfragment_id\n")
        for idx, fid in enumerate(fragment_ids):
            f.write(f"{idx}\t{fid}\n")

    print(f"[tdm] saved → {out_dir}")
    return result


# ── Step 4: SVD ──────────────────────────────────────────────────────────────

def build_svd(
    tdm: sparse.csr_matrix,
    k: int,
    n_iter: int,
    oversampling: int,
    random_state: int | None,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated randomized SVD and save artifacts."""
    print(f"[svd] computing randomized SVD (k={k}, n_iter={n_iter}) …")
    t0 = time.monotonic()

    U, sigma, Vt = randomized_svd(
        tdm,
        n_components=k,
        n_iter=n_iter,
        n_oversamples=oversampling,
        random_state=random_state,
    )

    wall_time = time.monotonic() - t0
    print(f"[svd] done in {wall_time:.1f}s  "
          f"U={U.shape}, σ={sigma.shape}, Vt={Vt.shape}")

    # Spectrum stats
    forb_sq = float(np.sum(tdm.data ** 2))
    evr = sigma ** 2 / forb_sq
    cum = np.cumsum(evr)
    eff90 = int(np.searchsorted(cum, 0.90) + 1) if len(cum) else k
    eff95 = int(np.searchsorted(cum, 0.95) + 1) if len(cum) else k

    stats = {
        "schema_version": "0.1.0",
        "input": {
            "matrix_shape": list(tdm.shape),
            "matrix_nnz": int(tdm.nnz),
            "matrix_density": round(tdm.nnz / (tdm.shape[0] * tdm.shape[1]), 8),
        },
        "params": {
            "k": k, "n_iter": n_iter,
            "oversampling": oversampling, "random_state": random_state,
        },
        "spectrum": {
            "singular_values": np.round(sigma, 1).tolist(),
            "explained_variance_ratio": np.round(evr, 4).tolist(),
            "cumulative_energy": np.round(cum, 4).tolist(),
            "energy_captured": round(float(cum[-1]), 4),
            "effective_rank_90": eff90,
            "effective_rank_95": eff95,
        },
        "sigma_summary": {
            "max": round(float(sigma.max()), 2),
            "min": round(float(sigma.min()), 2),
            "median": round(float(np.median(sigma)), 2),
            "mean": round(float(sigma.mean()), 3),
        },
        "runtime": {"wall_time_seconds": round(wall_time, 1)},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "U.npy", U)
    np.save(out_dir / "sigma.npy", sigma)
    np.save(out_dir / "Vt.npy", Vt)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[svd] energy captured: {cum[-1]:.2%},  "
          f"eff. rank 90%={eff90}, 95%={eff95}")
    print(f"[svd] saved → {out_dir}")
    return U, sigma, Vt


# ── Step 5: embeddings ───────────────────────────────────────────────────────

def build_embeddings(
    sigma: np.ndarray,
    Vt: np.ndarray,
    sigma_power: float,
    reshape_k: int | None,
    out_dir: Path,
) -> np.ndarray:
    """Build word vectors = Vt.T · diag(σ^α) and save artifacts."""
    k_full = len(sigma)
    if reshape_k is None:
        reshape_k = k_full

    print(f"[embeddings] V·Σ^{sigma_power}  (k={reshape_k}) …")

    if sigma_power == 0.0:
        word_vectors = Vt.T[:, :reshape_k]
    elif sigma_power == 1.0:
        word_vectors = Vt.T[:, :reshape_k] * sigma[np.newaxis, :reshape_k]
    else:
        word_vectors = (
            Vt.T[:, :reshape_k] * (sigma ** sigma_power)[np.newaxis, :reshape_k]
        )

    norms = np.linalg.norm(word_vectors, axis=1)
    stats = {
        "vocab_size": word_vectors.shape[0],
        "k": reshape_k,
        "sigma_power": sigma_power,
        "vector_norm_mean": round(float(norms.mean()), 4),
        "vector_norm_std": round(float(norms.std()), 4),
        "vector_norm_min": round(float(norms.min()), 6),
        "vector_norm_max": round(float(norms.max()), 4),
        "zero_norm_count": int((norms == 0).sum()),
    }
    meta = {
        "schema_version": "0.1.0",
        "method": "svd",
        "k": reshape_k,
        "sigma_power": sigma_power,
        "vocab_size": word_vectors.shape[0],
        "stats": stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "word_vectors.npy", word_vectors)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[embeddings] shape={word_vectors.shape}, "
          f"norm: mean={norms.mean():.4f} std={norms.std():.4f}")
    print(f"[embeddings] saved → {out_dir}")
    return word_vectors


# ── Step 6: postprocess (center + normalize) ─────────────────────────────────

def postprocess_center_norm(
    word_vectors: np.ndarray,
    sigma_power: float,
    reshape_k: int | None,
    out_dir: Path,
) -> np.ndarray:
    """Center and L2-normalize embeddings."""
    print("[postprocess] center + normalize …")
    t0 = time.monotonic()
    step_details = []

    # center
    mean_vec = word_vectors.mean(axis=0)
    centered = word_vectors - mean_vec
    step_details.append({
        "step": "center",
        "mean_vector_norm": round(float(np.linalg.norm(mean_vec)), 6),
    })

    # normalize
    norms_before = np.linalg.norm(centered, axis=1)
    normed = sk_normalize(centered, norm="l2", axis=1)
    step_details.append({
        "step": "normalize",
        "norms_before_mean": round(float(norms_before.mean()), 6),
        "norms_before_std": round(float(norms_before.std()), 6),
        "norms_before_min": round(float(norms_before.min()), 6),
        "norms_before_max": round(float(norms_before.max()), 6),
    })

    elapsed = time.monotonic() - t0
    meta = {
        "schema_version": "0.1.0",
        "method": "svd",
        "sigma_power": sigma_power,
        "reshape_k": reshape_k or word_vectors.shape[1],
        "steps": ["center", "normalize"],
        "shape": list(normed.shape),
        "elapsed_sec": round(elapsed, 2),
        "step_details": step_details,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "word_vectors.npy", normed)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[postprocess] shape={normed.shape}, elapsed={elapsed:.2f}s")
    print(f"[postprocess] saved → {out_dir}")
    return normed


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone embedding pipeline: corpus → vocab → TDM → SVD → embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="Path to pre-lemmatized corpus (one doc per line, sentences "
             "delimited by 2+ spaces)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root directory for all output artifacts",
    )
    # vocab
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.90)
    parser.add_argument("--min-tf", type=int, default=1)
    # fragments
    parser.add_argument(
        "--fragment-size", type=int, default=10,
        help="Sentences per fragment",
    )
    # tdm
    parser.add_argument(
        "--weighting", choices=["log-entropy", "tf-idf"], default="log-entropy",
    )
    # svd
    parser.add_argument("-k", type=int, default=300, help="SVD rank")
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--oversampling", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    # embeddings
    parser.add_argument("--sigma-power", type=float, default=0.5)
    parser.add_argument("--reshape-k", type=int, default=None)
    # workflow
    parser.add_argument(
        "--skip-until", choices=["vocab", "tdm", "svd", "embeddings", "postprocess"],
        default=None,
        help="Skip steps before this one (assumes artifacts already exist)",
    )

    args = parser.parse_args(argv)
    out = args.output_dir

    skip = args.skip_until
    total_t0 = time.monotonic()

    # ── vocab ──
    vocab_path = out / "vocab.tsv"
    if skip and skip != "vocab":
        print(f"[skip] vocab (loading from {vocab_path})")
        idx2word, word2idx, tf_arr, df_arr = _load_vocab_tsv(vocab_path)
        documents = _load_documents(args.corpus)
    else:
        idx2word, word2idx, tf_arr, df_arr, documents = build_vocab(
            args.corpus, args.min_df, args.max_df, args.min_tf,
        )
        save_vocab(vocab_path, idx2word, tf_arr, df_arr)

    vocab_size = len(idx2word)

    # ── fragments + tdm ──
    tdm_dir = out / "tdm"
    if skip and skip not in ("vocab", "tdm"):
        print(f"[skip] tdm (loading from {tdm_dir / 'matrix.npz'})")
        tdm = sparse.load_npz(tdm_dir / "matrix.npz")
    else:
        fragments, fragment_ids = build_fragments(
            documents, word2idx, args.fragment_size,
        )
        del documents  # free memory
        tdm = build_tdm(fragments, fragment_ids, vocab_size, args.weighting, tdm_dir)
        del fragments, fragment_ids

    # ── svd ──
    svd_dir = out / "svd"
    if skip and skip not in ("vocab", "tdm", "svd"):
        print(f"[skip] svd (loading from {svd_dir})")
        sigma = np.load(svd_dir / "sigma.npy")
        Vt = np.load(svd_dir / "Vt.npy")
    else:
        _, sigma, Vt = build_svd(
            tdm, args.k, args.n_iter, args.oversampling, args.random_state, svd_dir,
        )
    del tdm

    # ── embeddings ──
    emb_dir = out / "embeddings"
    if skip and skip not in ("vocab", "tdm", "svd", "embeddings"):
        print(f"[skip] embeddings (loading from {emb_dir})")
        word_vectors = np.load(emb_dir / "word_vectors.npy")
    else:
        word_vectors = build_embeddings(
            sigma, Vt, args.sigma_power, args.reshape_k, emb_dir,
        )
    del Vt

    # ── postprocess ──
    postprocess_center_norm(
        word_vectors, args.sigma_power, args.reshape_k,
        emb_dir / "word_vectors_center_norm",
    )

    total_elapsed = time.monotonic() - total_t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {total_elapsed:.1f}s")
    print(f"Artifacts: {out}")
    return 0


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_vocab_tsv(
    path: Path,
) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    from csv import DictReader

    idx2word: list[str] = []
    word2idx: dict[str, int] = {}
    tf_list, df_list = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in DictReader(f, delimiter="\t"):
            idx = int(row["idx"])
            idx2word.append(row["lemma"])
            word2idx[row["lemma"]] = idx
            tf_list.append(int(row["tf"]))
            df_list.append(int(row["df"]))
    return idx2word, word2idx, np.array(tf_list), np.array(df_list)


def _load_documents(corpus_path: Path) -> list[list[list[str]]]:
    documents: list[list[list[str]]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences = [
                s.strip().split()
                for s in re.split(r" {2,}", line)
                if s.strip()
            ]
            documents.append(sentences)
    return documents


if __name__ == "__main__":
    sys.exit(main())
