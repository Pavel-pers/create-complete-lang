"""
Convert a raw pre-lemmatized corpus (one doc per line, sentences delimited by 3+ spaces)
into the cclang DocLemma format (.lemma.json files in sharded directories).

Writes manifest.json and registers in DB.

Usage:
    python src/scripts/convert_raw_corpus.py \
        --corpus data/artifacts/archive/english_newlit_corpus.txt \
        --output-dir data/artifacts/corpus/en/newlit_corpus \
        --lang en \
        --version v1 \
        --method raw-lemmatized
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

# NER placeholder patterns
NER_TAGS = {"pron1", "person1", "ordinal1"}


def parse_line(line: str) -> list[list[dict]]:
    """Parse a corpus line into sentences of LemmaToken dicts."""
    sentences = []
    for sent_text in re.split(r" {3,}", line):
        sent_text = sent_text.strip()
        if not sent_text:
            continue
        tokens = []
        for word in sent_text.split():
            is_ne = word in NER_TAGS
            tokens.append({
                "token": word,
                "lemma": word,
                "is_ne": is_ne,
                "ner_result": word if is_ne else None,
                "is_oov": False,
                "is_ambiguous": False,
            })
        if tokens:
            sentences.append(tokens)
    return sentences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert raw corpus to DocLemma format")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--method", type=str, default="raw-lemmatized")
    parser.add_argument("--register-db", action="store_true",
                        help="Register build in DB")
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    total_docs = 0
    total_sentences = 0
    total_tokens = 0

    print(f"[convert] reading {args.corpus} ...")

    with open(args.corpus, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            sentences = parse_line(line)
            if not sentences:
                continue

            # Doc ID = sha256 of raw line
            doc_id = hashlib.sha256(line.encode("utf-8")).hexdigest()

            doc = {
                "schema_version": "0.1.0",
                "id": doc_id,
                "lang": args.lang,
                "sentences": sentences,
                "meta": {
                    "source": str(args.corpus.name),
                    "line_no": line_no,
                    "lemmatizer": args.method,
                    "stats": {
                        "num_sentences": len(sentences),
                        "num_tokens": sum(len(s) for s in sentences),
                    },
                },
            }

            # Sharded path: first 2 chars of hash
            shard = doc_id[:2]
            doc_dir = out_dir / shard
            doc_dir.mkdir(parents=True, exist_ok=True)
            doc_path = doc_dir / f"{doc_id}.lemma.json"

            with open(doc_path, "w", encoding="utf-8") as out_f:
                json.dump(doc, out_f, ensure_ascii=False)

            n_sent = len(sentences)
            n_tok = sum(len(s) for s in sentences)
            total_docs += 1
            total_sentences += n_sent
            total_tokens += n_tok

            if total_docs % 1000 == 0:
                print(f"  [{total_docs} docs, {total_tokens:,} tokens ...]")

    elapsed = time.monotonic() - t0
    print(f"[convert] {total_docs} docs, {total_sentences:,} sentences, "
          f"{total_tokens:,} tokens in {elapsed:.1f}s")

    # Write manifest.json
    manifest = {
        "schema_version": "0.2.0",
        "artifact_type": "corpus",
        "method": args.method,
        "version": args.version,
        "params": {
            "lang": args.lang,
            "source_file": str(args.corpus),
            "doc_count": total_docs,
            "sentence_count": total_sentences,
            "token_count": total_tokens,
        },
        "extra": {
            "ner_placeholders": sorted(NER_TAGS),
            "sentence_delimiter": "3+ spaces",
            "already_lemmatized": True,
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    print(f"[manifest] saved → {manifest_path}")

    # Register in DB
    if args.register_db:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        from cclang.io.db import get_conn
        from cclang.io.doc_state_store import DocStateStore
        from cclang.io.schemas import ProcessingStatus

        conn = get_conn(None)
        store = DocStateStore(conn)
        run_id = store.register_corpus_build(
            path=str(out_dir),
            version=args.version,
            status=ProcessingStatus.OK,
        )
        store.close()
        print(f"[db] registered corpus_build run_id={run_id}")

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
