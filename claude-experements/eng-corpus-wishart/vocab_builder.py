"""
Build and filter vocabulary from corpus documents.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

# Minimal English stop words (lemmatized forms)
STOP_WORDS = frozenset({
    "the", "a", "an", "be", "have", "do", "say", "go", "get", "make",
    "can", "will", "would", "could", "should", "may", "might", "shall",
    "not", "no", "nor", "yet", "so", "but", "and", "or", "if", "then",
    "than", "that", "this", "these", "those", "it", "its",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "too", "very",
    "just", "also", "now", "about", "up", "down",
})

_NUMERIC_RE = re.compile(r"^\d+$")


def build_vocab(
    documents: list[list[list[str]]],
    min_df: int = 3,
    max_df_ratio: float = 0.85,
    min_tf: int = 5,
    min_len: int = 2,
    remove_stopwords: bool = True,
    remove_numeric: bool = True,
) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    """
    Build filtered vocabulary from corpus documents.

    Parameters
    ----------
    documents : list of docs, each doc = list of sentences, each sentence = list of lemmas
    min_df : minimum document frequency
    max_df_ratio : maximum document frequency as ratio of total docs
    min_tf : minimum term frequency
    min_len : minimum token length
    remove_stopwords : remove English stop words
    remove_numeric : remove purely numeric tokens

    Returns
    -------
    idx2word, word2idx, tf_array, df_array
    """
    print("[vocab] counting term/document frequencies ...")
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()

    for doc in documents:
        doc_lemmas: set[str] = set()
        for sent in doc:
            for lemma in sent:
                tf_counter[lemma] += 1
                doc_lemmas.add(lemma)
        for lemma in doc_lemmas:
            df_counter[lemma] += 1

    n_docs = len(documents)
    max_df_abs = int(n_docs * max_df_ratio)

    print(f"[vocab] documents: {n_docs}, raw vocab: {len(tf_counter)}")
    print(f"[vocab] filtering: min_tf={min_tf}, min_df={min_df}, "
          f"max_df={max_df_abs} ({max_df_ratio:.0%}), min_len={min_len}")

    filtered = sorted(
        (
            (lemma, tf_counter[lemma], df_counter[lemma])
            for lemma in tf_counter
            if tf_counter[lemma] >= min_tf
            and df_counter[lemma] >= min_df
            and df_counter[lemma] <= max_df_abs
            and len(lemma) >= min_len
            and (not remove_stopwords or lemma not in STOP_WORDS)
            and (not remove_numeric or not _NUMERIC_RE.match(lemma))
        ),
        key=lambda x: -x[1],
    )

    idx2word = [item[0] for item in filtered]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    tf_arr = np.array([item[1] for item in filtered])
    df_arr = np.array([item[2] for item in filtered])

    print(f"[vocab] filtered vocab: {len(idx2word)}")
    return idx2word, word2idx, tf_arr, df_arr


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


def load_vocab(path: Path) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    idx2word: list[str] = []
    tf_list, df_list = [], []
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            idx2word.append(parts[1])
            tf_list.append(int(parts[2]))
            df_list.append(int(parts[3]))
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return idx2word, word2idx, np.array(tf_list), np.array(df_list)
