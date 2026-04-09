#!/usr/bin/env python3
"""Upload cluster-explore artifacts to S3 via StorageManager boto3 client."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eng-corpus-wishart"))

from storage_setup import make_storage

S3_PREFIX = "artifacts/experiments/eng-corpus-cluster-explore"
HERE = Path(__file__).resolve().parent


def upload_all():
    storage = make_storage()
    s3 = storage._cloud.client
    bucket = storage._cloud._cfg.bucket
    root_prefix = storage._cloud._cfg.root_prefix.as_posix().strip("/")

    upload_list: list[tuple[Path, str]] = []

    # Top-level files
    for name in [
        "cluster_explore_phase1.ipynb",
        "cluster_explore_phase2.ipynb",
        "helpers.py",
        "build_notebook.py",
        "claude-report.md",
        "task.md",
        "cluster_analysis_guide.md",
        "versions/iterations.md",
    ]:
        p = HERE / name
        if p.exists():
            upload_list.append((p, f"{root_prefix}/{S3_PREFIX}/{name}"))

    # Figures (walk subdirectories: phase1/, phase2/)
    fig_dir = HERE / "figures"
    if fig_dir.exists():
        for f in fig_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(HERE)
                upload_list.append((f, f"{root_prefix}/{S3_PREFIX}/{rel.as_posix()}"))

    # Cache (walk subdirectories: phase1/, phase2/)
    cache_dir = HERE / "cache"
    if cache_dir.exists():
        for f in cache_dir.rglob("*"):
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
