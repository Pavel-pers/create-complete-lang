from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl

SCHEMA_VERSION = "0.1.0"

# ---------- DataBase --------

class FetchedItem(BaseModel):
    url: HttpUrl
    sha256: str
    local_path: str
    ts: str = Field(default_factory= lambda: datetime.now().isoformat() + 'Z')

# ---------- Manifest --------

class FetchSatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"

class FetchManifestRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    url: HttpUrl
    status: FetchSatus
    status_code: Optional[int] = None
    sha: Optional[str] = None
    size: Optional[int] = None
    local_path: Optional[str] = None
    error: Optional[str] = None
    ts: str = Field(default_factory= lambda: datetime.now().isoformat() + 'Z')

    STATUS_OK = FetchSatus.OK
    STATUS_ERROR = FetchSatus.ERROR
    STATUS_SKIPPED = FetchSatus.SKIPPED

    @property
    def id(self):
        return self.sha

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


class CorpusBookLink(BaseModel):
    id: str
    link: str
    meta: Optional[Dict] = None


class SourcePDF(BaseModel):
    """Single record in data/sources/*.jsonl"""
    url: HttpUrl
    lang: str = Field(default="mr", description="ISO 639-1 language code")
    source: Optional[str] = Field(default=None, description="Origin site or collection")
    added_at: datetime = Field(default_factory=datetime.utcnow)


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
