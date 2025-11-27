import hashlib
import json
import sqlite3
from pathlib import Path

from cclang.common.logx import get_logger
from cclang.ingest.net import DownloadResult
from cclang.io.schemas import SourcePDF
from cclang.ingest.fs import get_shard_path
from pipelines.corpus.fetch_pdfs import run_pipeline


def test_run_pipeline_reuses_existing_file_and_updates_db(monkeypatch, tmp_path: Path):
    url = "http://example.com/book.pdf"
    file_bytes = b"book-bytes"
    file_sha = hashlib.sha256(file_bytes).hexdigest()

    output_base = tmp_path / "output"
    manifest_path = tmp_path / "manifest.jsonl"
    db_path = tmp_path / "db.sqlite"
    urls_path = tmp_path / "urls.jsonl"

    # Precreate shard path to simulate already downloaded file
    shard_path = get_shard_path(output_base, file_sha, ".pdf")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(file_bytes)

    # Temp file that fake downloader will "produce"
    temp_file = tmp_path / "temp" / "file.part"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_bytes(file_bytes)

    def fake_download(file_url: str):
        assert file_url == url
        return DownloadResult(status=200, size_bytes=len(file_bytes), tmp_file=temp_file, sha256=file_sha)

    monkeypatch.setattr("pipelines.corpus.fetch_pdfs.download_file_to_temp", fake_download)

    with open(urls_path, "w", encoding="utf-8") as f:
        f.write(SourcePDF(url=url, lang="mr", source="epustakalay").model_dump_json() + "\n")

    log = get_logger("test")
    run_pipeline(urls_path=urls_path, output_base_path=output_base, manifest_path=manifest_path,
                 database_path=db_path, log=log, max_count=1)

    # Temp file must be removed because shard already existed
    assert not temp_file.exists()
    # Manifest should contain a single OK record pointing to the shard path
    manifest_records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    assert len(manifest_records) == 1
    record = manifest_records[0]
    assert record["status"] == "ok"
    assert Path(record["local_path"]) == shard_path
    # DB should have the mapping url -> sha -> path
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT url, sha256, local_path FROM fetch_items")
    rows = cur.fetchall()
    conn.close()
    assert rows == [(url, file_sha, str(shard_path))]
