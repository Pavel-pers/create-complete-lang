import sys
import types
import hashlib
import json
from pathlib import Path

# Some environments may miss optional deps; stub them so imports work.
if "contourpy" not in sys.modules:
    sys.modules["contourpy"] = types.SimpleNamespace(max_threads=lambda: 0)

if "indicnlp" not in sys.modules:
    indicnlp_pkg = types.ModuleType("indicnlp")
    tokenize_pkg = types.ModuleType("indicnlp.tokenize")
    sentence_mod = types.SimpleNamespace(sentence_split=lambda text, lang=None: text.split("."))
    indic_mod = types.SimpleNamespace(trivial_tokenize=lambda text, lang=None: text.split())
    tokenize_pkg.sentence_tokenize = sentence_mod
    tokenize_pkg.indic_tokenize = indic_mod
    sys.modules["indicnlp"] = indicnlp_pkg
    sys.modules["indicnlp.tokenize"] = tokenize_pkg
    sys.modules["indicnlp.tokenize.sentence_tokenize"] = sentence_mod
    sys.modules["indicnlp.tokenize.indic_tokenize"] = indic_mod

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
    psycopg_stub = types.SimpleNamespace(connect=lambda *args, **kwargs: None, Connection=object)
    sys.modules["psycopg"] = psycopg_stub

from cclang.common.logx import get_logger
from cclang.config.s3 import S3Config
from cclang.io.fs import get_shard_relative
from cclang.io.schemas import DocRaw, DocTok, PdfState, ProcessingStatus
from pipelines.corpus import tokenize_text


def test_run_pipeline_writes_output_and_updates_state(monkeypatch, tmp_path: Path):
    data_base = tmp_path / "data"
    output_base_rel = Path("tokenized")
    manifest_path = data_base / "manifests" / "pl_tokenize_text.jsonl"

    # Prepare input text file
    text_rel = Path("raw/sample.jsonl")
    text_abs = data_base / text_rel
    text_abs.parent.mkdir(parents=True, exist_ok=True)
    sample_sentences = ["हे एक वाक्य आहे.", "ही दुसरी ओळ आहे."]
    with open(text_abs, "w", encoding="utf-8") as f:
        for idx, sentence in enumerate(sample_sentences, start=1):
            f.write(
                DocRaw(
                    id=f"doc#{idx}",
                    text=sentence,
                    meta={"page_num": idx},
                ).model_dump_json(ensure_ascii=False)
                + "\n"
            )

    pdf_sha = hashlib.sha256(b"sample-pdf").hexdigest()
    tasks = [
        PdfState(
            pdf_sha=pdf_sha,
            pdf_path="ignored.pdf",
            text_sha="textsha",
            text_path=str(text_rel),
            text_status=ProcessingStatus.OK,
        )
    ]
    updated_calls: list[tuple] = []

    class FakePdfStateStore:
        def __init__(self, _conn):
            self._closed = False

        def filter_by_status(self, text_status=None, tokenize_status=None):
            return tasks

        def update_tokenize_status(self, pdf_sha, tokenize_status, tokenize_sha, tokenize_path, ts):
            updated_calls.append((pdf_sha, tokenize_status, tokenize_sha, tokenize_path, ts))

        def close(self):
            self._closed = True

    # Disable real DB and S3
    monkeypatch.setattr(tokenize_text, "PdfStateStore", FakePdfStateStore)
    monkeypatch.setattr(tokenize_text, "get_conn", lambda _: None)
    monkeypatch.setattr(
        tokenize_text,
        "load_s3_config",
        lambda: S3Config(enable=False, bucket="", root_prefix=Path(""), region=None, access_key=None, secret_key=None),
    )

    log = get_logger("test")

    tokenize_text.run_pipeline(
        output_base_path=output_base_rel,
        database_dsn="postgresql://example",
        manifest_path=manifest_path,
        log=log,
        target_status=None,
        max_count=None,
        data_path=data_base,
    )

    # Output file exists and matches manifest path
    output_files = list((data_base / output_base_rel).rglob("*.tok.json"))
    assert len(output_files) == 1
    output_file = output_files[0]
    expected_rel = output_base_rel / get_shard_relative(pdf_sha, ".tok.json")
    assert output_file.relative_to(data_base) == expected_rel

    # Manifest contains a single OK record with correct sha and path
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(manifest_records) == 1
    record = manifest_records[0]
    content = output_file.read_text(encoding="utf-8")
    expected_sha = hashlib.sha256(content.encode()).hexdigest()
    assert record["status"] == "ok"
    assert record["tokenize_sha"] == expected_sha
    assert Path(record["tokenize_path"]) == expected_rel

    # DB status updated with path and sha
    assert updated_calls == [
        (pdf_sha, ProcessingStatus.OK, expected_sha, str(expected_rel), record["ts"])
    ]

    # Saved DocTok is valid and has sentences
    parsed = DocTok.model_validate_json(content)
    assert len(parsed.sentences) >= 1
