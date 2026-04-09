"""
CBOW (Continuous Bag of Words) embedding method via gensim Word2Vec.
"""
from __future__ import annotations

import time

import numpy as np
from gensim.models import Word2Vec


def build(
    documents: list[list[list[str]]],
    word2idx: dict[str, int],
    vocab_size: int,
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 10,
    sg: int = 0,
    do_center: bool = True,
    do_normalize: bool = True,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """
    Build CBOW embeddings using gensim Word2Vec.

    Parameters
    ----------
    sg : 0 = CBOW, 1 = Skip-gram
    """
    t0 = time.monotonic()

    # Flatten documents to list of sentences (each sentence = list of tokens)
    # Filter to only tokens in word2idx
    sentences = []
    for doc in documents:
        for sent in doc:
            filtered = [tok for tok in sent if tok in word2idx]
            if filtered:
                sentences.append(filtered)

    print(f"[cbow] training Word2Vec (sg={sg}, dim={vector_size}, "
          f"window={window}, epochs={epochs}, sentences={len(sentences)}) ...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=seed,
    )

    train_time = time.monotonic() - t0
    print(f"[cbow] training done in {train_time:.1f}s, model vocab: {len(model.wv)}")

    # Extract vectors in word2idx order
    idx2word = sorted(word2idx.keys(), key=lambda w: word2idx[w])
    word_vectors = np.zeros((vocab_size, vector_size), dtype=np.float32)
    found = 0
    for i, word in enumerate(idx2word):
        if word in model.wv:
            word_vectors[i] = model.wv[word]
            found += 1

    print(f"[cbow] mapped {found}/{vocab_size} words")

    # Postprocess
    from sklearn.preprocessing import normalize as sk_normalize
    if do_center:
        word_vectors -= word_vectors.mean(axis=0)
    if do_normalize:
        word_vectors = sk_normalize(word_vectors, norm="l2", axis=1)

    norms = np.linalg.norm(word_vectors, axis=1)
    total_time = time.monotonic() - t0

    meta = {
        "method": "cbow" if sg == 0 else "skipgram",
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "epochs": epochs,
        "sg": sg,
        "seed": seed,
        "postprocess": {
            "center": do_center,
            "normalize": do_normalize,
        },
        "training_sentences": len(sentences),
        "model_vocab_size": len(model.wv),
        "mapped_words": found,
        "embedding_stats": {
            "shape": list(word_vectors.shape),
            "norm_mean": round(float(norms.mean()), 6),
            "norm_std": round(float(norms.std()), 6),
            "norm_min": round(float(norms.min()), 6),
            "norm_max": round(float(norms.max()), 6),
            "zero_norm_count": int((norms == 0).sum()),
        },
        "total_time_sec": round(total_time, 1),
    }

    return word_vectors, meta
