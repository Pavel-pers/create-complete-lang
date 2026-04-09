#!/usr/bin/env python3
"""
Upload experiment artifacts to S3.

Uploads all embedding versions and cluster versions to:
  s3://cclang-cloud/cclang/data/artifacts/experiments/eng-corpus-wishart/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from storage_setup import make_storage, EXPERIMENT_ROOT

# S3 target prefix (relative to StorageManager root: cclang/data/)
S3_EXPERIMENT_PREFIX = "artifacts/experiments/eng-corpus-wishart"


def upload_all() -> None:
    storage = make_storage()
    s3_client = storage._cloud.client
    bucket = storage._cloud._cfg.bucket
    # After init, root_prefix = cclang/data
    root_prefix = storage._cloud._cfg.root_prefix.as_posix().strip("/")

    exp_data = EXPERIMENT_ROOT / "data"

    upload_list: list[tuple[Path, str]] = []

    # Embeddings
    emb_root = exp_data / "embeddings"
    for version_dir in sorted(emb_root.iterdir()) if emb_root.exists() else []:
        if not version_dir.is_dir():
            continue
        for f in version_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(exp_data)  # embeddings/v1/word_vectors.npy
                s3_key = f"{root_prefix}/{S3_EXPERIMENT_PREFIX}/{rel.as_posix()}"
                upload_list.append((f, s3_key))

    # Clusters
    clu_root = exp_data / "clusters"
    for cluster_dir in sorted(clu_root.iterdir()) if clu_root.exists() else []:
        if not cluster_dir.is_dir():
            continue
        for f in cluster_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(exp_data)
                s3_key = f"{root_prefix}/{S3_EXPERIMENT_PREFIX}/{rel.as_posix()}"
                upload_list.append((f, s3_key))

    # Validation
    val_root = exp_data / "validation"
    for val_dir in sorted(val_root.iterdir()) if val_root.exists() else []:
        if not val_dir.is_dir():
            continue
        for f in val_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(exp_data)
                s3_key = f"{root_prefix}/{S3_EXPERIMENT_PREFIX}/{rel.as_posix()}"
                upload_list.append((f, s3_key))

    # Reports and code (top-level files)
    for name in [
        "claude-report.md",
        "claude-clustering-review.md",
        "task.md",
        "label_saving_sample.py",
        "versions/iterations.md",
    ]:
        f = EXPERIMENT_ROOT / name
        if f.is_file():
            s3_key = f"{root_prefix}/{S3_EXPERIMENT_PREFIX}/{name}"
            upload_list.append((f, s3_key))

    print(f"Uploading {len(upload_list)} files to s3://{bucket}/{root_prefix}/{S3_EXPERIMENT_PREFIX}/")
    print()

    total_size = 0
    for i, (local_path, s3_key) in enumerate(upload_list, 1):
        size = local_path.stat().st_size
        total_size += size
        s3_client.upload_file(str(local_path), bucket, s3_key)
        if i % 10 == 0 or i == len(upload_list):
            print(f"  {i}/{len(upload_list)} uploaded ({total_size / 1024 / 1024:.1f} MB)")

    print()
    print(f"Done. Total: {len(upload_list)} files, {total_size / 1024 / 1024:.1f} MB")
    print(f"Location: s3://{bucket}/{root_prefix}/{S3_EXPERIMENT_PREFIX}/")

    storage.close()


if __name__ == "__main__":
    upload_all()
