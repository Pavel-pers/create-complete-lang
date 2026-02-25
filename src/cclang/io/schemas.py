from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl

SCHEMA_VERSION = "0.1.0"


# ---------- DataBase --------

class FetchedItem(BaseModel):
    url: HttpUrl
    sha256: str
    local_path: str
    ts: str = Field(default_factory=lambda: datetime.now().isoformat() + 'Z')


class ProcessingStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    NOT_PROCESSED = "pending"
    RUNNING = "running"


# ---------- Normalized DB task/result models --------

class OcrTask(BaseModel):
    doc_id: str
    pdf_path: str


class TokenizeTask(BaseModel):
    doc_id: str
    text_path: str


class LemmatizeTask(BaseModel):
    doc_id: str
    tokenize_path: str


class LemmaResult(BaseModel):
    doc_id: str
    method: str
    artefact_id: Optional[str] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.OK)
    path: Optional[str] = None
    updated_at: Optional[str] = None


class OcrResult(BaseModel):
    doc_id: str
    artefact_id: Optional[str] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.OK)
    path: Optional[str] = None
    updated_at: Optional[str] = None


class VocabParams(BaseModel):
    min_df: Optional[int] = None
    max_df: Optional[int] = None
    min_tf: Optional[int] = None
    replace_ner: Optional[bool] = False
    normalize: Optional[bool] = False
    ignore_oov: Optional[bool] = False


class VocabStats(BaseModel):
    total_lemmas: int
    total_documents: int


class VocabInfo(BaseModel):
    run_id: int
    lemma_method: str
    artefact_id: Optional[str] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.OK)
    path: Optional[str] = None
    vocab_params: Optional[VocabParams] = None
    vocab_stats: Optional[VocabStats] = None
    updated_at: Optional[str] = None


class CorpusInfo(BaseModel):
    run_id: int
    vocab_id: int
    fragment_size: int
    status: ProcessingStatus = Field(default=ProcessingStatus.OK)
    stats: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None


class TdmInfo(BaseModel):
    run_id: int
    corpus_id: int
    weighting: Optional[str] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.OK)
    path: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None

class SvdBuildInfo(BaseModel):
    run_id: int
    tdm_id: int
    k: int
    params: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    path: Optional[str] = None
    status: ProcessingStatus = Field(default=ProcessingStatus.OK)
    updated_at: Optional[str] = None


class SvdInputInfo(BaseModel):
    """Snapshot if input data"""
    tdm_build_id: int
    matrix_shape: List[int] = Field(..., min_length=2, max_length=2)
    matrix_nnz: int
    matrix_density: float


class SvdParams(BaseModel):
    """Hyperparams for SVD."""
    k: int
    n_iter: int = 5
    oversampling: int = 10
    random_state: Optional[int] = None


class SvdSpectrum(BaseModel):
    """Spectral characteristics for SVD."""
    singular_values: List[float]
    explained_variance_ratio: List[float]
    cumulative_energy: List[float]
    energy_captured: float
    effective_rank_90: int
    effective_rank_95: int


class SvdSigmaSummary(BaseModel):
    max: float
    min: float
    median: float
    mean: float


class SvdRuntime(BaseModel):
    wall_time_seconds: float


class SvdBuildStats(BaseModel):
    """Full statistic of SVD-build, for json dumping"""
    schema_version: str = Field(default=SCHEMA_VERSION)
    input: SvdInputInfo
    params: SvdParams
    spectrum: SvdSpectrum
    sigma_summary: SvdSigmaSummary
    runtime: SvdRuntime
    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    ts: str = Field(default_factory=lambda: datetime.now().isoformat() + 'Z')

    STATUS_OK: ClassVar[FetchSatus] = FetchSatus.OK
    STATUS_ERROR: ClassVar[FetchSatus] = FetchSatus.ERROR
    STATUS_SKIPPED: ClassVar[FetchSatus] = FetchSatus.SKIPPED

    @property
    def id(self) -> str:
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
    ts: str = Field(default_factory=lambda: datetime.now().isoformat() + 'Z')

    STATUS_OK: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.OK
    STATUS_ERROR: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.ERROR
    STATUS_SKIPPED: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.SKIPPED

    @property
    def id(self) -> str:
        return self.pdf_sha


class TokenizeManifestRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    pdf_sha: str

    text_path: str
    status: ProcessedPdfStatus

    tokenize_path: Optional[str]
    tokenize_sha: Optional[str]
    error: Optional[str] = None

    ts: str = Field(default_factory=lambda: datetime.now().isoformat() + 'Z')

    STATUS_OK: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.OK
    STATUS_ERROR: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.ERROR
    STATUS_SKIPPED: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.SKIPPED

    @property
    def id(self) -> str:
        return self.pdf_sha


class LemmatizeManifestRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    pdf_sha: str

    tokenize_path: str
    status: ProcessedPdfStatus

    lemma_path: Optional[str]
    lemma_sha: Optional[str]
    error: Optional[str] = None

    ts: str = Field(default_factory=lambda: datetime.now().isoformat() + 'Z')

    STATUS_OK: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.OK
    STATUS_ERROR: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.ERROR
    STATUS_SKIPPED: ClassVar[ProcessedPdfStatus] = ProcessedPdfStatus.SKIPPED

    @property
    def id(self) -> str:
        return self.pdf_sha


class UploadStatus(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    SUCCEDED = "succeeded"
    FAILED = "failed"


class UploadManifestRecord(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)

    local_path: Path
    cloud_key: str
    status: UploadStatus

    ts: str = Field(default_factory=lambda: datetime.now().isoformat() + 'Z')

    @property
    def id(self) -> str:
        return str(self.local_path) + ';' + str(self.cloud_key)


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


class LemmaToken(BaseModel):
    token: str
    lemma: str
    pos: Optional[str] = None

    is_ne: bool = False
    ner_result: Optional[str] = None

    analyses: Optional[List[str]] = None
    # for example: ["lemma<n><pl>", "lemma2<v><past>"]
    is_oov: bool = False
    is_ambiguous: bool = False


class DocLemma(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    id: str
    lang: str
    sentences: List[List[LemmaToken]]
    meta: Optional[Dict[str, Any]] = None


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
class FragmentProcessed(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    id: str
    source_doc_id: str
    position: int
    token_ids: list[int]
    vocab_id: int


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
