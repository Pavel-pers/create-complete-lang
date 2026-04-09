"""
Read English DocLemma corpus from S3 via StorageManager.

Provides an iterator over documents, yielding lists of sentences
(each sentence = list of lemma strings).
"""
from __future__ import annotations

import gzip
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from cclang.core.storage import StorageManager


# Relative to CloudConfig.base_path (= "data/"), so no "data/" prefix
CORPUS_S3_PREFIX = "artifacts/corpus/en/newlit_corpus"

NER_PLACEHOLDERS = frozenset({"pron1", "person1", "ordinal1"})


def list_corpus_keys(storage: StorageManager) -> list[str]:
    """
    List all .lemma.json.gz keys in the corpus prefix via S3 paginator.
    Returns relative keys (without the root_prefix).
    """
    s3_client = storage._cloud.client
    bucket = storage._cloud._cfg.bucket
    # After StorageManager init, root_prefix = original_prefix / cloud_base_path
    # e.g. "cclang" / "data" = "cclang/data"
    effective_prefix = storage._cloud._cfg.root_prefix.as_posix().strip("/")

    # Full S3 listing prefix: effective_prefix + corpus relative path
    full_prefix = f"{effective_prefix}/{CORPUS_S3_PREFIX}/" if effective_prefix else f"{CORPUS_S3_PREFIX}/"

    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".lemma.json.gz"):
                # Strip the effective prefix (cclang/data/) to get relative key
                if effective_prefix and key.startswith(effective_prefix + "/"):
                    relative = key[len(effective_prefix) + 1:]
                else:
                    relative = key
                keys.append(relative)
    return keys


def iter_corpus_docs(
    storage: StorageManager,
    keys: list[str] | None = None,
    remove_ner: bool = True,
    progress_every: int = 500,
) -> Iterator[list[list[str]]]:
    """
    Iterate over corpus documents, yielding doc = list of sentences,
    each sentence = list of lemma strings.

    Parameters
    ----------
    storage : StorageManager
    keys : list of relative S3 keys; if None, will call list_corpus_keys()
    remove_ner : if True, skip NER placeholder tokens
    progress_every : print progress every N documents
    """
    if keys is None:
        keys = list_corpus_keys(storage)

    for i, key in enumerate(keys):
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[corpus] read {i + 1}/{len(keys)} docs")

        # Strip .gz suffix so StorageManager auto-decompresses
        open_path = key[:-3] if key.endswith(".gz") else key
        with storage.open(Path(open_path), mode="r") as f:
            doc_data = json.load(f)

        sentences: list[list[str]] = []
        for sent_tokens in doc_data["sentences"]:
            lemmas = []
            for tok in sent_tokens:
                lemma = tok["lemma"]
                if remove_ner and tok.get("is_ne", False):
                    continue
                lemmas.append(lemma)
            if lemmas:
                sentences.append(lemmas)

        if sentences:
            yield sentences


def _read_one_doc(
    storage: StorageManager,
    key: str,
    remove_ner: bool,
) -> list[list[str]] | None:
    """Read a single document (for parallel loading)."""
    open_path = key[:-3] if key.endswith(".gz") else key
    try:
        with storage.open(Path(open_path), mode="r") as f:
            doc_data = json.load(f)
    except Exception:
        return None

    sentences: list[list[str]] = []
    for sent_tokens in doc_data["sentences"]:
        lemmas = []
        for tok in sent_tokens:
            lemma = tok["lemma"]
            if remove_ner and tok.get("is_ne", False):
                continue
            lemmas.append(lemma)
        if lemmas:
            sentences.append(lemmas)

    return sentences if sentences else None


def load_all_docs(
    storage: StorageManager,
    keys: list[str] | None = None,
    remove_ner: bool = True,
    max_workers: int = 16,
) -> list[list[list[str]]]:
    """Load all corpus documents into memory using parallel downloads."""
    if keys is None:
        keys = list_corpus_keys(storage)

    print(f"[corpus] loading {len(keys)} documents (workers={max_workers}) ...")
    docs: list[list[list[str]]] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_read_one_doc, storage, key, remove_ner): key
            for key in keys
        }
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"[corpus] {done}/{len(keys)} docs loaded")
            result = future.result()
            if result is not None:
                docs.append(result)

    print(f"[corpus] loaded {len(docs)} documents total")
    return docs
