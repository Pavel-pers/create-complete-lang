import hashlib
import json
from pathlib import Path

from cclang.common.logx import get_logger
from cclang.ingest.net import DownloadResult
from cclang.io.schemas import SourcePDF
from cclang.io.fs import get_shard_path
from pipelines.corpus.fetch_pdfs import run_pipeline


class InMemoryCursor:
    def __init__(self, storage: list[tuple[str, str, str, str]]):
        self._storage = storage
        self._result: list[tuple[str, str, str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, query: str, params=None):
        params = tuple(params or [])
        normalized = " ".join(query.lower().split())
        if "from fetch_items where url" in normalized:
            url = params[0]
            self._result = [row for row in self._storage if row[0] == url]
        elif "from fetch_items where sha256" in normalized:
            sha = params[0]
            self._result = [row for row in self._storage if row[1] == sha]
        elif normalized.startswith("insert into fetch_items"):
            self._storage.append((params[0], params[1], params[2], params[3]))
            self._result = []
        else:
            self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class InMemoryConnection:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []
        self.closed = False

    def cursor(self, *_, **__):
        return InMemoryCursor(self.rows)

    def commit(self):
        return None

    def close(self):
        self.closed = True


def test_run_pipeline_reuses_existing_file_and_updates_db(monkeypatch, tmp_path: Path):
    url = "http://example.com/book.pdf"
    file_bytes = b"book-bytes"
    file_sha = hashlib.sha256(file_bytes).hexdigest()

    data_base = tmp_path / "data"
    output_base_rel = Path("artifacts/raw_pdfs")
    output_base_abs = data_base / output_base_rel
    manifest_path = data_base / "manifests" / "manifest.jsonl"
    urls_path = tmp_path / "urls.jsonl"
    db_conn = InMemoryConnection()

    # Precreate shard path to simulate already downloaded file
    shard_path_abs = get_shard_path(output_base_abs, file_sha, ".pdf")
    shard_path_abs.parent.mkdir(parents=True, exist_ok=True)
    shard_path_abs.write_bytes(file_bytes)
    shard_path_rel = output_base_rel / shard_path_abs.relative_to(output_base_abs)

    # Temp file that fake downloader will "produce"
    temp_file = tmp_path / "temp" / "file.part"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_bytes(file_bytes)

    def fake_download(file_url: str, *, sm=None, download_folder=None):
        assert file_url == url
        assert sm is not None
        return DownloadResult(
            status=200,
            size_bytes=len(file_bytes),
            tmp_file=temp_file,
            sha256=file_sha,
        )

    monkeypatch.setattr("pipelines.corpus.fetch_pdfs.download_file_to_temp", fake_download)
    monkeypatch.setattr("pipelines.corpus.fetch_pdfs.get_conn", lambda _: db_conn)

    with open(urls_path, "w", encoding="utf-8") as f:
        f.write(SourcePDF(url=url, lang="mr", source="epustakalay").model_dump_json() + "\n")

    log = get_logger("test")
    run_pipeline(
        urls_path=urls_path,
        output_base_path=output_base_rel,
        manifest_path=manifest_path,
        database_dsn="postgresql://example",
        log=log,
        max_count=1,
        data_path=data_base,
    )

    # Temp file must be removed because shard already existed
    assert not temp_file.exists()
    # Manifest should contain a single OK record pointing to the shard path
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
    ]
    assert len(manifest_records) == 1
    record = manifest_records[0]
    assert record["status"] == "ok"
    assert Path(record["local_path"]) == shard_path_rel
    # File still exists on disk at the absolute location
    assert shard_path_abs.exists()
    # DB should have the mapping url -> sha -> path
    assert [(row[0], row[1], row[2]) for row in db_conn.rows] == [(url, file_sha, str(shard_path_rel))]
