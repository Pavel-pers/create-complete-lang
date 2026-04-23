#!/usr/bin/env python3
"""Upload eng-corpus-cluster-gaps artifacts to S3 via StorageManager boto3 client."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eng-corpus-wishart"))

from storage_setup import make_storage

S3_PREFIX = "artifacts/experiments/eng-corpus-cluster-gaps"
HERE = Path(__file__).resolve().parent


def upload_all() -> None:
    storage = make_storage()
    s3 = storage._cloud.client
    bucket = storage._cloud._cfg.bucket
    root_prefix = storage._cloud._cfg.root_prefix.as_posix().strip("/")

    upload_list: list[tuple[Path, str]] = []

    top_files = [
        "gaps_analysis.ipynb",
        "helpers.py",
        "alignment.py",
        "rulsif_utils.py",
        "gap_search.py",
        "wordnet_utils.py",
        "benchmarks.py",
        "build_notebook.py",
        "upload_to_s3.py",
        "claude-report.md",
        "task.md",
        "plan.md",
        "versions/iterations.md",
    ]
    for name in top_files:
        p = HERE / name
        if p.exists():
            upload_list.append((p, f"{root_prefix}/{S3_PREFIX}/{name}"))

    for subdir in ["figures", "cache", "results", "benchmarks"]:
        d = HERE / subdir
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(HERE)
                    upload_list.append((f, f"{root_prefix}/{S3_PREFIX}/{rel.as_posix()}"))

    print(f"Uploading {len(upload_list)} files to s3://{bucket}/{root_prefix}/{S3_PREFIX}/")
    total = 0
    for i, (local, key) in enumerate(upload_list, 1):
        s3.upload_file(str(local), bucket, key)
        total += local.stat().st_size
        if i % 10 == 0 or i == len(upload_list):
            print(f"  {i}/{len(upload_list)} ({total/1024/1024:.1f} MB)")

    print(f"Done: {total/1024/1024:.1f} MB total")
    storage.close()


if __name__ == "__main__":
    upload_all()
