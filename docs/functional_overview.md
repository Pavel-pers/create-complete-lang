# Functional Overview

This project revolves around a small set of reusable components for ingesting PDF corpora, tracking what was fetched, and mirroring artifacts to S3/Yandex Object Storage. Below is a map of the main functionality and how to use it.

## Core utilities

- **Logging** (`cclang.common.logx`): Console/JSON logging with `BoundLogger.bind(**extra)` to enrich records. `setup_logging()` configures stdout/stderr handlers and optional JSON file rotation.
- **S3 configuration** (`cclang.config.s3.S3Config`): Populated from env (`CCLANG_S3_ENABLE`, `CCLANG_S3_BUCKET`, `CCLANG_S3_ROOT_PREFIX`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
- **Local/cloud file manager** (`cclang.ingest.fs.FileManager`): Creates temp files, shards paths by checksum, stores locally, and mirrors to cloud via `cloud_upload/cloud_download`.
- **S3 store** (`cclang.io.cloud.S3Store`): Thin wrapper around `boto3.client` with configurable upload threads, connection pool, retries on transient network errors, and async upload queue callbacks.
- **Network ingest** (`cclang.ingest.net.download_file_to_temp`): Downloads a URL to a temp file, returns size/sha256/status, and cleans up on failure.
- **Manifest** (`cclang.io.manifest.ManifestStore`): Thread-safe append-only JSONL manifest writer/reader for pipeline outputs.
- **Fetched items DB** (`cclang.io.fetched_items_store.FetchedItemsStore` + `cclang.io.db.ensure_schema`): SQLite-backed registry of downloaded files keyed by URL/sha256 to avoid duplicates.
- **Task queue logging** (`cclang.models.tasks_queue.TaskQueue`): `queue.Queue` subclass that periodically logs remaining tasks.

## Pipelines and scripts

- **Discover links** (`pipelines.corpus.discover_links` → `cclang-discover-links`): Scrapes epustakalay book links and emits `SourcePDF` records.
- **Fetch PDFs** (`pipelines.corpus.fetch_pdfs` → `cclang-fetch-pdfs`): Reads URLs, downloads PDFs with worker threads, writes manifests, updates SQLite registry, and optionally mirrors to S3.
- **Upload local data to S3** (`scripts.upload_local_to_s3` → `upload-local-to-s3`): Walks selected `data/` subdirectories and uploads them to S3 with configurable threads/connection pool and optional skip-existing.

## Data layout (default)

- `data/raw_pdfs/` — raw downloads.  
- `data/corpora/`, `data/artifacts/` — processed outputs.  
- `data/temp/` — temporary files used during downloads and uploads.

## CLI quick reference

```bash
# Discover links
cclang-discover-links -path data/sources/links.jsonl --log-level INFO

# Fetch PDFs into data/ and mirror to S3 (env must contain S3 creds)
cclang-fetch-pdfs --urls data/sources/links.jsonl --data-path data --log-level INFO

# Upload local data subdirs to S3
upload-local-to-s3 --dir raw_pdfs --dir corpora --no-skip-existing \\
  --upload-threads 16 --log-level INFO
```

Adjust `--upload-threads` to match available bandwidth/CPU; the S3 client pool defaults to 4×threads (min 32). Use `--no-skip-existing` to overwrite objects instead of issuing HEAD checks.
