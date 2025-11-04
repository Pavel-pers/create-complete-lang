from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


SCHEMA_VERSION = "0.1.0"


# ---------- Corpus ----------

class DocRaw(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    id: str
    text: str
    meta: Optional[Dict] = None


class DocTok(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    id: str
    lang: str
    sentences: List[List[str]]  # tokenized sentences
    meta: Optional[Dict] = None


# ---------- Vocabulary / Matrix ----------

class TokenInfo(BaseModel):
    freq: int
    df: Optional[int] = None
    extra: Optional[Dict] = None


class Vocab(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    tokens: Dict[str, TokenInfo]


# ---------- Vectors (LSA/SVD) ----------

class VectorsMeta(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    lang: str
    k: int
    method: str = "svd"
    explained_variance: List[float] = Field(default_factory=list)  # per-component variance (not ratio)
    explained_variance_ratio: List[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    params: Dict = Field(default_factory=dict)


# ---------- Graph ----------

class GraphMeta(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    lang: str
    build: str  # e.g., "knn", "threshold"
    params: Dict = Field(default_factory=dict)
    stats: Dict = Field(default_factory=dict)