from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl

SCHEMA_VERSION = "0.1.0"

# ---------- DataBase --------

class FetchedItem(BaseModel):
    url: HttpUrl
    sha256: str
    local_path: str
    ts: str = Field(default_factory= lambda: datetime.now().isoformat() + 'Z')

class ProcessingStatus(str, Enum):
    OK = "ok"
    ERROR = "error"

class PdfState(BaseModel):
    pdf_sha: str
    pdf_path: str

    text_sha: Optional[str] = None
    text_path: Optional[str] = None
    text_status: Optional[ProcessingStatus] = None
    text_updated_at: Optional[str] = None

    tokenize_sha: Optional[str] = None
    tokenize_path: Optional[str] = None
    tokenize_status: Optional[ProcessingStatus] = None
    tokenize_updated_at: Optional[str] = None

    lemma_sha: Optional[str] = None
    lemma_path: Optional[str] = None
    lemma_status: Optional[ProcessingStatus] = None
    lemma_updated_at: Optional[str] = None

    PROCESSING_STATUS_OK: ClassVar[ProcessingStatus] = ProcessingStatus.OK
    PROCESSING_STATUS_ERROR: ClassVar[ProcessingStatus] = ProcessingStatus.ERROR


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

    STATUS_OK: ClassVar[FetchSatus] = FetchSatus.OK
    STATUS_ERROR: ClassVar[FetchSatus] = FetchSatus.ERROR
    STATUS_SKIPPED: ClassVar[FetchSatus] = FetchSatus.SKIPPED

    @property
    def id(self):
        # Fallback to URL when sha is unknown (errors)
        return self.sha or str(self.url)

class ProcessedPdfStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"

class ProcessedPdfManifestRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    pdf_path: str
    pdf_sha: str
    status: ProcessedPdfStatus

    text_path: Optional[str] = None
    text_sha: Optional[str] = None
    error: Optional[str] = None
    ts: str = Field(default_factory= lambda: datetime.now().isoformat() + 'Z')

    STATUS_OK: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.OK
    STATUS_ERROR: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.ERROR
    STATUS_SKIPPED: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.SKIPPED

    @property
    def id(self):
        return self.pdf_sha


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
