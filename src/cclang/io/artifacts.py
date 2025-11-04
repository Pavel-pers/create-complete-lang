from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

from .schemas import (
    GraphMeta,
    VectorsMeta,
    Vocab,
)


# ------ Matrices ------

def load_cooc(path: str) -> sparse.spmatrix:
    return sparse.load_npz(path)


def save_cooc(path: str, M: sparse.spmatrix) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, M)


# ------ Vocabulary ------

def load_vocab(path: str) -> Vocab:
    with open(path, "r", encoding="utf-8") as f:
        return Vocab.model_validate_json(f.read())


def save_vocab(path: str, vocab: Vocab) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(vocab.model_dump_json(indent=2))


# ------ Vectors ------

def load_vectors(npy_path: str, meta_path: str) -> Tuple[np.ndarray, VectorsMeta]:
    vecs = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = VectorsMeta.model_validate_json(f.read())
    return vecs, meta


def save_vectors(npy_path: str, meta_path: str, vecs: np.ndarray, meta: VectorsMeta) -> None:
    Path(npy_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, vecs)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.model_dump_json(indent=2))


# ------ Graph ------

def load_graph(path: str) -> nx.Graph:
    return nx.read_graphml(path)


def save_graph(path: str, G: nx.Graph) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, path)


def load_graph_meta(path: str) -> GraphMeta:
    with open(path, "r", encoding="utf-8") as f:
        return GraphMeta.model_validate_json(f.read())


def save_graph_meta(path: str, meta: GraphMeta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(meta.model_dump_json(indent=2))
