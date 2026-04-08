# English NewLit Corpus

## Overview

| Metric | Value |
|---|---|
| Language | en |
| Documents | 11,046 (10,885 unique by SHA-256) |
| Sentences | 16,086,966 |
| Tokens | 240,322,097 |
| Unique tokens | 275,668 |
| Source | `data/artifacts/archive/english_newlit_corpus.txt` (41.6 MB) |
| Preprocessing | Pre-lemmatized, NER-replaced |

## Artifact

**DB**: `corpus_builds` run_id=1, version=v1, status=ok

**Path**: `data/artifacts/corpus/en/newlit_corpus/`

**Storage**: local + S3 (`cclang/data/artifacts/corpus/en/newlit_corpus/`)

| | Files | Size |
|---|---|---|
| Compressed (.gz) | 10,885 | ~1 GB |
| Uncompressed | 10,885 | ~25 GB |

### Directory structure

```
data/artifacts/corpus/en/newlit_corpus/
  manifest.json                          # ArtifactManifest (не сжат)
  <shard>/                               # 2-char hex prefix от SHA-256
    <sha256>.lemma.json.gz               # DocLemma (gzip)
```

## Document format (DocLemma)

Each file is a gzip-compressed JSON following the `DocLemma` schema:

```json
{
  "schema_version": "0.1.0",
  "id": "<sha256 of raw text line>",
  "lang": "en",
  "sentences": [
    [
      {"token": "person1", "lemma": "person1", "is_ne": true,  "ner_result": "person1", "is_oov": false, "is_ambiguous": false},
      {"token": "be",      "lemma": "be",      "is_ne": false, "ner_result": null,      "is_oov": false, "is_ambiguous": false},
      {"token": "upon",    "lemma": "upon",     "is_ne": false, "ner_result": null,      "is_oov": false, "is_ambiguous": false}
    ],
    [ ... ]
  ],
  "meta": {
    "source": "english_newlit_corpus.txt",
    "line_no": 7413,
    "lemmatizer": "raw-lemmatized",
    "stats": {
      "num_sentences": 5,
      "num_tokens": 74
    }
  }
}
```

### LemmaToken fields

| Field | Type | Description |
|---|---|---|
| `token` | str | Surface form (= lemma, corpus already lemmatized) |
| `lemma` | str | Lemma (= token) |
| `is_ne` | bool | `true` for NER placeholders |
| `ner_result` | str? | Placeholder tag if `is_ne`, else `null` |
| `is_oov` | bool | Always `false` (corpus is pre-built) |
| `is_ambiguous` | bool | Always `false` |

### NER placeholders

| Tag | Meaning | Frequency |
|---|---|---|
| `pron1` | Pronouns (he, she, it, I...) | 30,651,946 |
| `person1` | Person names | 8,564,914 |
| `ordinal1` | Ordinal numerals (first, second) | 1,945,090 |

## Manifest

```json
{
  "schema_version": "0.2.0",
  "artifact_type": "corpus",
  "method": "raw-lemmatized",
  "version": "v1",
  "params": {
    "lang": "en",
    "source_file": "data/artifacts/archive/english_newlit_corpus.txt",
    "doc_count": 11046,
    "sentence_count": 16086966,
    "token_count": 240322097
  },
  "extra": {
    "ner_placeholders": ["ordinal1", "person1", "pron1"],
    "sentence_delimiter": "3+ spaces",
    "already_lemmatized": true
  }
}
```

## Source format

Raw file: one line per document, tokens space-separated, sentences delimited by 3+ spaces. Empty lines (6) skipped. Text already lemmatized (`was/is/are` -> `be`, etc.).

## Conversion

Script: `src/scripts/convert_raw_corpus.py`

```bash
python src/scripts/convert_raw_corpus.py \
  --corpus data/artifacts/archive/english_newlit_corpus.txt \
  --output-dir data/artifacts/corpus/en/newlit_corpus \
  --lang en --version v1 --method raw-lemmatized --register-db
```

Document ID = SHA-256 of the raw text line. 161 documents with duplicate hashes were overwritten (11,046 lines -> 10,885 unique files).
