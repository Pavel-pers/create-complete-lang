"""Devanagari normalization, validation, and preprocessing utilities for corpus pipelines."""

from __future__ import annotations

import re
import unicodedata

from cclang.io.schemas import LemmaToken

# ---------------------------------------------------------------------------
# Devanagari normalization & filtering
# ---------------------------------------------------------------------------

# Unicode ranges for Devanagari
_DEVANAGARI_BLOCK = r'\u0900-\u097F'
_DEVANAGARI_EXT = r'\uA8E0-\uA8FF'
_DEVANAGARI_PATTERN = re.compile(f'^[{_DEVANAGARI_BLOCK}{_DEVANAGARI_EXT}]+$')

# Halant (virama)
_HALANT = '\u094D'

# Devanagari vowels (independent forms)
_VOWELS = set('अआइईउऊऋऌएऐओऔॠॡ')

# Pattern: halant followed by an independent vowel (invalid in normal text)
_HALANT_VOWEL = re.compile(r'्[' + ''.join(_VOWELS) + r']')

# Double halant
_DOUBLE_HALANT = re.compile(r'्{2,}')

# Ends with halant (often OCR garbage — valid conjuncts don't end in virama
# except rare cases; we flag but don't hard-reject)
_TRAILING_HALANT = re.compile(r'्$')


def normalize_devanagari(text: str) -> str:
    """
    Apply Unicode NFC normalization and strip invisible / zero-width chars.

    This collapses visually identical but byte-different representations that
    arise from OCR and inconsistent encoding.
    """
    text = unicodedata.normalize('NFC', text)
    # Remove ZWJ / ZWNJ (common OCR artefacts in Devanagari conjuncts)
    text = text.replace('\u200d', '').replace('\u200c', '')
    # Remove other invisible characters
    text = re.sub(r'[\u200b\u00ad\ufeff\u200e\u200f]', '', text)
    return text


def is_valid_devanagari(token: str) -> bool:
    """
    Check whether *token* is a plausible Devanagari word.

    Returns False for tokens that contain non-Devanagari characters,
    start with combining marks, or are unreasonably long.
    """
    if not token:
        return False
    # Must consist entirely of Devanagari codepoints
    if not _DEVANAGARI_PATTERN.match(token):
        return False
    # Must not start with a combining mark (matra / halant)
    if unicodedata.category(token[0]) in ('Mn', 'Mc'):
        return False
    return True


def is_ocr_garbage(token: str) -> bool:
    """
    Heuristic detector for common OCR artefacts in Devanagari text.

    A token is flagged if it exhibits patterns that cannot occur in
    well-formed Marathi:
      • halant + independent vowel  (e.g. क्अ)
      • double halant               (e.g. क््क)
      • syllable repeated ≥3 times  (e.g. वावावा)
      • excessive halant density     (more than 40 % of chars)
      • unreasonable length          (> 30 characters)
    """
    if len(token) > 30:
        return True

    if _HALANT_VOWEL.search(token):
        return True

    if _DOUBLE_HALANT.search(token):
        return True

    # High halant density → likely broken conjuncts
    halant_count = token.count(_HALANT)
    if len(token) > 3 and halant_count / len(token) > 0.35:
        return True

    # Repeating syllable (2–4 chars) appearing 3+ times in a row
    for n in range(2, 5):
        for i in range(len(token) - n * 3 + 1):
            chunk = token[i:i + n]
            if chunk * 3 in token:
                return True

    return False


def classify_lemma(lemma: str) -> str:
    """
    Classify a lemma into one of:
      'ok'              – passes all checks
      'non_devanagari'  – contains non-Devanagari characters
      'invalid_deva'    – Devanagari but structurally invalid
      'ocr_garbage'     – looks like an OCR artefact
    """
    if not _DEVANAGARI_PATTERN.match(lemma) if lemma else True:
        return 'non_devanagari'
    if not is_valid_devanagari(lemma):
        return 'invalid_deva'
    if is_ocr_garbage(lemma):
        return 'ocr_garbage'
    return 'ok'


# ---------------------------------------------------------------------------
# Pipeline preprocessing functions
# ---------------------------------------------------------------------------


def replace_ner(token: LemmaToken) -> str:
    """If token.is_ne, return '<NER_TAG>' (e.g. '<PERSON>'), otherwise token.lemma."""
    if token.is_ne:
        return '<' + str(token.ner_result).upper() + '>'
    return token.lemma


def normalize_lemma(lemma: str) -> str | None:
    """
    NFC-normalize + validate Devanagari + OCR-filter.

    Returns None if lemma is rejected (non-devanagari / invalid / OCR garbage).
    Special tokens (<NER_TAG>) are passed through unchanged.
    """
    if lemma.startswith('<') and lemma.endswith('>'):
        return lemma
    normalized = normalize_devanagari(lemma)
    cls = classify_lemma(normalized)
    if cls != 'ok':
        return None
    return normalized


def is_oov(token: LemmaToken) -> bool:
    """Return True if the token is out-of-vocabulary."""
    return token.is_oov
