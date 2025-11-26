from __future__ import annotations

import io
import json
import re
import unicodedata
from typing import Iterable, Iterator, Union
import zstandard as zstd

from .schemas import DocRaw, DocTok, BaseModel


def _open_text_read(path: str):
    """Open text (possibly zst-compressed) for reading as text."""
    if path.endswith(".zst"):
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _open_text_write(path: str):
    """Open text (possibly zst-compressed) for writing as text."""
    if path.endswith(".zst"):
        fh = open(path, "wb")
        cctx = zstd.ZstdCompressor(level=10)
        stream = cctx.stream_writer(fh)
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def iter_docs(path: str) -> Iterator[BaseModel]:
    """Stream validated DocRaw/DocTok objects from .jsonl or .jsonl.zst."""
    with _open_text_read(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "sentences" in obj:
                yield DocTok.model_validate(obj)
            else:
                yield DocRaw.model_validate(obj)


def write_docs(path: str, docs: Iterable[BaseModel]) -> None:
    """Write validated DocRaw/DocTok to .jsonl or .jsonl.zst (one JSON per line)."""
    with _open_text_write(path) as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")


# ---------------- Normalization & Tokenization ----------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize_text(
        text: str,
        *,
        strip_html: bool = True,
        nfkc: bool = True,
        lower: bool = False,
) -> str:
    """Minimal, language-agnostic normalization."""
    if strip_html:
        text = _HTML_TAG_RE.sub(" ", text)
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if lower:
        text = text.lower()
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _simple_whitespace_tokenize(text: str) -> list[list[str]]:
    """Very simple sentence/token split: split by [.!?] then whitespace."""
    # sentence split (naive)
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # tokens: split on whitespace, strip basic punctuation
        toks = [re.sub(r"^\W+|\W+$", "", t) for t in s.split()]
        toks = [t for t in toks if t]
        if toks:
            out.append(toks)
    return out


def tokenize(text: str, lang: str) -> DocTok:
    """
    Tokenizer stub with a placeholder for Marathi.
    For now: identical simple whitespace tokenizer for all langs.
    (Later: plug-in real tokenizers per language.)
    """
    raise NotImplementedError()
    sentences = _simple_whitespace_tokenize(text)
    return DocTok(id="NA", lang=lang, sentences=sentences, meta=None)


def batch_tokenize(
        input_path: str,
        output_path: str,
        lang: str,
        *,
        assume_raw: bool = True,
) -> None:
    """
    Read DocRaw or raw lines (assume_raw=True), produce DocTok .jsonl(.zst).
    If input contains DocRaw JSONL: it will use their id/meta.
    If raw lines: ids are incremental.
    """
    counter = 0
    out_docs = []
    if assume_raw:
        with _open_text_read(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = DocRaw(id=f"{lang}-{counter}", text=normalize_text(line), meta=None)
                counter += 1
                dtok = tokenize(raw.text, lang=lang)
                dtok.id = raw.id
                dtok.meta = raw.meta
                out_docs.append(dtok)
    else:
        for doc in iter_docs(input_path):
            if isinstance(doc, DocRaw):
                text = normalize_text(doc.text)
                dtok = tokenize(text, lang=lang)
                dtok.id = doc.id
                dtok.meta = doc.meta
            else:
                dtok = doc
            out_docs.append(dtok)

    write_docs(output_path, out_docs)
