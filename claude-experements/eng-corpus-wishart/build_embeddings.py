#!/usr/bin/env python3
"""
CLI: Build word embeddings from English corpus.

Usage:
    python build_embeddings.py --method svd-lsa --version v1 --weighting log-entropy --svd-k 300
    python build_embeddings.py --method svd-lsa --version v2 --weighting ppmi --svd-k 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from storage_setup import make_storage, make_doc_store, EXPERIMENT_DATA
from corpus_reader import list_corpus_keys, load_all_docs
from vocab_builder import build_vocab, save_vocab, load_vocab
from cclang.io.schemas import ProcessingStatus


EMBEDDING_METHODS = {}


def register_method(name: str):
    def decorator(fn):
        EMBEDDING_METHODS[name] = fn
        return fn
    return decorator


@register_method("svd-lsa")
def _build_svd_lsa(documents, word2idx, vocab_size, args):
    from embeddings.svd_lsa import build
    return build(
        documents=documents,
        word2idx=word2idx,
        vocab_size=vocab_size,
        weighting=args.weighting,
        svd_k=args.svd_k,
        sigma_power=args.sigma_power,
        fragment_size=args.fragment_size,
        do_center="center" in args.postprocess,
        do_normalize="normalize" in args.postprocess,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )


@register_method("cbow")
def _build_cbow(documents, word2idx, vocab_size, args):
    from embeddings.cbow import build
    return build(
        documents=documents,
        word2idx=word2idx,
        vocab_size=vocab_size,
        vector_size=args.svd_k,  # reuse --svd-k as vector dim
        window=args.window,
        epochs=args.epochs,
        sg=0,
        do_center="center" in args.postprocess,
        do_normalize="normalize" in args.postprocess,
    )


@register_method("skipgram")
def _build_skipgram(documents, word2idx, vocab_size, args):
    from embeddings.cbow import build
    return build(
        documents=documents,
        word2idx=word2idx,
        vocab_size=vocab_size,
        vector_size=args.svd_k,
        window=args.window,
        epochs=args.epochs,
        sg=1,
        do_center="center" in args.postprocess,
        do_normalize="normalize" in args.postprocess,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build word embeddings from English corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # General
    parser.add_argument("--method", required=True, choices=list(EMBEDDING_METHODS.keys()),
                        help="Embedding method")
    parser.add_argument("--version", required=True, help="Artifact version (e.g. v1, v2)")

    # Vocab filtering
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df-ratio", type=float, default=0.85)
    parser.add_argument("--min-tf", type=int, default=5)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--no-remove-ner", action="store_true")
    parser.add_argument("--no-remove-stopwords", action="store_true")
    parser.add_argument("--no-remove-numeric", action="store_true")

    # SVD-LSA specific
    parser.add_argument("--weighting", default="log-entropy",
                        choices=["log-entropy", "tf-idf", "ppmi"])
    parser.add_argument("--svd-k", type=int, default=300)
    parser.add_argument("--sigma-power", type=float, default=0.5)
    parser.add_argument("--fragment-size", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    # CBOW/Skip-gram specific
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window size")
    parser.add_argument("--epochs", type=int, default=10, help="Word2Vec training epochs")

    # Postprocessing
    parser.add_argument("--postprocess", nargs="+", default=["center", "normalize"],
                        choices=["center", "normalize", "none"])

    # Workflow
    parser.add_argument("--skip-corpus-load", action="store_true",
                        help="Skip corpus load, use cached vocab")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip DB registration")

    args = parser.parse_args(argv)

    if "none" in args.postprocess:
        args.postprocess = []

    out_dir = EXPERIMENT_DATA / "embeddings" / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = out_dir / "vocab.tsv"

    print(f"{'='*60}")
    print(f"Building embeddings: method={args.method}, version={args.version}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    total_t0 = time.monotonic()

    # ── Storage ──
    storage = make_storage()

    # ── DB registration ──
    build_id = None
    if not args.no_db:
        try:
            doc_store = make_doc_store()
            build_id = doc_store.register_embedding_build(
                path=str(out_dir),
                version=args.version,
                status=ProcessingStatus.RUNNING,
            )
            print(f"[db] registered embedding_build run_id={build_id}")
        except Exception as e:
            print(f"[db] WARNING: could not register build: {e}")
            build_id = None

    try:
        # ── Corpus + Vocab ──
        if args.skip_corpus_load and vocab_path.exists():
            print("[skip] loading cached vocab")
            idx2word, word2idx, tf_arr, df_arr = load_vocab(vocab_path)
            # Still need documents for building TDM
            print("[corpus] loading documents from S3 ...")
            documents = load_all_docs(
                storage,
                remove_ner=not args.no_remove_ner,
            )
        else:
            print("[corpus] loading documents from S3 ...")
            documents = load_all_docs(
                storage,
                remove_ner=not args.no_remove_ner,
            )
            idx2word, word2idx, tf_arr, df_arr = build_vocab(
                documents,
                min_df=args.min_df,
                max_df_ratio=args.max_df_ratio,
                min_tf=args.min_tf,
                min_len=args.min_len,
                remove_stopwords=not args.no_remove_stopwords,
                remove_numeric=not args.no_remove_numeric,
            )
            save_vocab(vocab_path, idx2word, tf_arr, df_arr)

        vocab_size = len(idx2word)

        # ── Build embeddings ──
        build_fn = EMBEDDING_METHODS[args.method]
        word_vectors, meta = build_fn(documents, word2idx, vocab_size, args)

        del documents  # free memory

        # ── Save artifacts ──
        np.save(out_dir / "word_vectors.npy", word_vectors)

        # Build full manifest
        manifest = {
            "schema_version": "0.2.0",
            "artifact_type": "embedding",
            "method": args.method,
            "version": args.version,
            "params": {
                "vocab_size": vocab_size,
                "min_df": args.min_df,
                "max_df_ratio": args.max_df_ratio,
                "min_tf": args.min_tf,
                "remove_ner": not args.no_remove_ner,
                "remove_stopwords": not args.no_remove_stopwords,
                "remove_numeric": not args.no_remove_numeric,
                **{k: v for k, v in meta.items() if k != "_hierarchy"},
            },
            "created_at": datetime.now().isoformat() + "Z",
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

        total_elapsed = time.monotonic() - total_t0
        print(f"\n{'='*60}")
        print(f"Embeddings built in {total_elapsed:.1f}s")
        print(f"Artifacts: {out_dir}")
        print(f"Vocab: {vocab_size}, Shape: {word_vectors.shape}")

        # ── Complete DB ──
        if build_id is not None:
            try:
                doc_store.complete_embedding_build(build_id, ProcessingStatus.OK)
                print(f"[db] completed embedding_build run_id={build_id}")
            except Exception as e:
                print(f"[db] WARNING: could not complete build: {e}")

    except Exception:
        if build_id is not None:
            try:
                doc_store.complete_embedding_build(build_id, ProcessingStatus.ERROR)
            except Exception:
                pass
        raise
    finally:
        storage.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
