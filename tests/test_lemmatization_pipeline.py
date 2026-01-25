import sys
import types
import hashlib
import json
from pathlib import Path

# Stub optional deps so imports succeed in minimal test envs.
if "contourpy" not in sys.modules:
    sys.modules["contourpy"] = types.SimpleNamespace(max_threads=lambda: 0)

if "botocore" not in sys.modules:
    botocore_pkg = types.ModuleType("botocore")
    exceptions_mod = types.ModuleType("botocore.exceptions")

    class _DummyExc(Exception):
        pass

    exceptions_mod.ClientError = type("ClientError", (_DummyExc,), {"__init__": lambda self, response, operation_name=None: None})
    exceptions_mod.ConnectTimeoutError = type("ConnectTimeoutError", (_DummyExc,), {})
    exceptions_mod.ConnectionClosedError = type("ConnectionClosedError", (_DummyExc,), {})
    exceptions_mod.EndpointConnectionError = type("EndpointConnectionError", (_DummyExc,), {})
    exceptions_mod.ReadTimeoutError = type("ReadTimeoutError", (_DummyExc,), {})
    sys.modules["botocore"] = botocore_pkg
    sys.modules["botocore.exceptions"] = exceptions_mod

if "botocore.config" not in sys.modules:
    config_mod = types.ModuleType("botocore.config")

    class _DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    config_mod.Config = _DummyConfig
    sys.modules["botocore.config"] = config_mod

if "boto3" not in sys.modules:
    sys.modules["boto3"] = types.SimpleNamespace(client=lambda *args, **kwargs: None)

if "psycopg" not in sys.modules:
    sql_mod = types.SimpleNamespace(SQL=lambda arg: str(arg))
    psycopg_stub = types.SimpleNamespace(connect=lambda *args, **kwargs: None, Connection=object, sql=sql_mod)
    sys.modules["psycopg"] = psycopg_stub
    sys.modules["psycopg.sql"] = sql_mod

if "apertium" not in sys.modules:
    apertium_mod = types.ModuleType("apertium")

    class _DummyModeNotInstalled(Exception):
        pass

    class _DummyAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, token):
            return []

    apertium_mod.ModeNotInstalled = _DummyModeNotInstalled
    apertium_mod.Analyzer = _DummyAnalyzer
    sys.modules["apertium"] = apertium_mod

from cclang.common.logx import get_logger
from cclang.config.s3 import S3Config
from cclang.ingest.fs import get_shard_relative
from cclang.io.schemas import DocTok, DocLemma, LemmaToken, PdfState, ProcessingStatus
from pipelines.corpus import lemmatization


def test_run_pipeline_writes_output_and_updates_state(monkeypatch, tmp_path: Path):
    data_base = tmp_path / "data"
    output_base_rel = Path("lemmatized")
    manifest_path = data_base / "manifests" / "pl_lemmatization.jsonl"

    # Prepare tokenized input
    tokenize_rel = Path("tokenized/sample.tok.json")
    tokenize_abs = data_base / tokenize_rel
    tokenize_abs.parent.mkdir(parents=True, exist_ok=True)
    sample_tokens = [["घर", "आहे"], ["हे", "परीक्षण"]]
    doc_tok = DocTok(id="doc#1", lang="mr", sentences=sample_tokens)
    tokenize_abs.write_text(doc_tok.model_dump_json(ensure_ascii=False), encoding="utf-8")

    pdf_sha = hashlib.sha256(b"sample-pdf").hexdigest()
    tasks = [
        PdfState(
            pdf_sha=pdf_sha,
            pdf_path="ignored.pdf",
            text_status=ProcessingStatus.OK,
            tokenize_status=ProcessingStatus.OK,
            tokenize_path=str(tokenize_rel),
        )
    ]
    updated_calls: list[tuple] = []

    class FakePdfStateStore:
        def __init__(self, _conn):
            self._closed = False

        def filter_by_status(self, text_status=None, tokenize_status=None, lemma_status=None):
            return tasks

        def update_lemma_status(self, pdf_sha, lemma_status, lemma_sha, lemma_path, ts):
            updated_calls.append((pdf_sha, lemma_status, lemma_sha, lemma_path, ts))

        def close(self):
            self._closed = True

    def fake_lemmatize(token: str) -> LemmaToken:
        return LemmaToken(token=token, lemma=f"{token}_lemma")

    # Disable real DB/S3 and heavy lemmatizer
    monkeypatch.setattr(lemmatization, "PdfStateStore", FakePdfStateStore)
    monkeypatch.setattr(lemmatization, "get_conn", lambda _: None)
    monkeypatch.setattr(
        lemmatization,
        "load_s3_config",
        lambda: S3Config(
            enable=False, bucket="", root_prefix=Path(""), region=None, access_key=None, secret_key=None
        ),
    )
    monkeypatch.setattr(lemmatization, "lemmatize_marathi_token", fake_lemmatize)

    log = get_logger("test")

    lemmatization.run_pipeline(
        output_base_path=output_base_rel,
        database_dsn="postgresql://example",
        manifest_path=manifest_path,
        log=log,
        target_status=None,
        max_count=None,
        data_path=data_base,
    )

    output_files = list((data_base / output_base_rel).rglob("*.lemma.json"))
    assert len(output_files) == 1
    output_file = output_files[0]
    expected_rel = output_base_rel / get_shard_relative(pdf_sha, ".lemma.json")
    assert output_file.relative_to(data_base) == expected_rel

    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(manifest_records) == 1
    record = manifest_records[0]
    content = output_file.read_text(encoding="utf-8")
    expected_sha = hashlib.sha256(content.encode()).hexdigest()
    assert record["status"] == "ok"
    assert record["lemma_sha"] == expected_sha
    assert Path(record["lemma_path"]) == expected_rel
    assert record["tokenize_path"] == str(tokenize_rel)

    assert updated_calls == [
        (pdf_sha, ProcessingStatus.OK, expected_sha, str(expected_rel), record["ts"])
    ]

    parsed = DocLemma.model_validate_json(content)
    assert parsed.sentences
    assert all(tok.lemma.endswith("_lemma") for sentence in parsed.sentences for tok in sentence)
