# src/cclang/text/lemmatizers/apertium_marathi.py
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Optional, Tuple, List

import apertium
from apertium import Analyzer

_APERTIUM_LOGGER = logging.getLogger("apertium")
# Silence noisy subprocess warnings from the upstream library.
_APERTIUM_LOGGER.setLevel(logging.ERROR)
_APERTIUM_LOGGER.propagate = False
if not _APERTIUM_LOGGER.handlers:
    _APERTIUM_LOGGER.addHandler(logging.NullHandler())
apertium.logger = _APERTIUM_LOGGER

from cclang.io.schemas import LemmaToken


class LemmatizerError(RuntimeError):
    """Base class for lemmatizer-related errors."""


class ApertiumModeNotInstalledError(LemmatizerError):
    """Raised when the requested Apertium mode (language module) is not installed."""


class ApertiumRuntimeError(LemmatizerError):
    """Raised when Apertium fails during analysis (subprocess / IO / unexpected output)."""


_THREAD_LOCAL = threading.local()


def _get_thread_local_analyzer(lang: str) -> Analyzer:
    """
    Create one Analyzer per thread to avoid shared-state issues and reduce init overhead.
    """
    if not hasattr(_THREAD_LOCAL, "analyzers"):
        _THREAD_LOCAL.analyzers = {}

    analyzers = _THREAD_LOCAL.analyzers
    if lang not in analyzers:
        try:
            analyzers[lang] = Analyzer(lang)
        except apertium.ModeNotInstalled as exc:
            raise ApertiumModeNotInstalledError(
                f"Apertium mode '{lang}' is not installed. "
                f"Install the language module (e.g. apertium-installer / system packages)."
            ) from exc
        except Exception as exc:
            raise ApertiumRuntimeError(f"Failed to initialize Apertium Analyzer('{lang}')") from exc

    return analyzers[lang]


def _reading_to_str(reading) -> str:
    """
    Convert one reading (list of subreadings) into a compact 'lemma<tag><tag>+...' string.
    """
    parts: List[str] = []
    for sub in reading:
        tag_str = "".join(f"<{t}>" for t in (sub.tags or []))
        parts.append(f"{sub.baseform}{tag_str}")
    return "+".join(parts)


def _extract_pos_from_reading(reading) -> Optional[str]:
    """
    POS heuristic: use the first tag of the first subreading (often closer to the lexical root).
    """
    if not reading:
        return None
    first = reading[0]
    if not getattr(first, "tags", None):
        return None
    return first.tags[0] if first.tags else None


def _strip_oov_marker(lemma: str) -> str:
    """
    Streamparser uses '*' to denote unknown analyses.
    We normalize '*lemma' -> 'lemma'.
    """
    return lemma[1:] if lemma.startswith("*") else lemma


@lru_cache(maxsize=200_000)
def _lemmatize_cached(lang: str, token: str) -> Tuple[str, Optional[str], Tuple[str, ...], bool, bool]:
    """
    Cached core lemmatization returning pure-python primitives (safe to cache).
    Returns:
      (lemma, pos, analyses_tuple, is_oov, is_ambiguous)
    """
    if not token:
        return "", None, tuple(), True, False

    analyzer = _get_thread_local_analyzer(lang)

    try:
        lexical_units = analyzer.analyze(token)
    except Exception as exc:
        raise ApertiumRuntimeError(f"Apertium analyze() failed for token={token!r}") from exc

    if not lexical_units:
        # Defensive fallback
        return token, None, tuple(), True, False

    # Apertium may emit extra units (e.g., sentence boundary like ./.<sent> in examples),
    # so try to select the unit corresponding to the original token.
    lu = None
    for u in lexical_units:
        if getattr(u, "wordform", None) == token:
            lu = u
            break
    if lu is None:
        lu = lexical_units[0]

    readings = getattr(lu, "readings", None) or []
    knownness = getattr(lu, "knownness", None)
    knownness_symbol = getattr(knownness, "symbol", "")  # streamparser unknown uses '*' :contentReference[oaicite:2]{index=2}

    analyses: List[str] = []
    lemma_candidates: List[str] = []
    pos_candidates: List[Optional[str]] = []

    for r in readings:
        if not r:
            continue
        analyses.append(_reading_to_str(r))
        lemma_candidates.append(_strip_oov_marker(getattr(r[0], "baseform", "") or ""))
        pos_candidates.append(_extract_pos_from_reading(r))

    # OOV: either explicit unknown symbol '*' or no usable readings.
    is_oov = (knownness_symbol == "*") or (len(lemma_candidates) == 0)

    if is_oov:
        # If analyzer has no reading, keep token as lemma.
        return token, None, tuple(analyses), True, False

    # Pick the first lemma as the canonical one (simple, deterministic baseline).
    lemma = lemma_candidates[0] if lemma_candidates[0] else token
    pos = pos_candidates[0] if pos_candidates else None

    # Ambiguity: multiple DISTINCT lemma candidates.
    is_ambiguous = len(set(l for l in lemma_candidates if l)) > 1

    return lemma, pos, tuple(analyses), is_oov, is_ambiguous


def lemmatize_marathi_token(token: str) -> LemmaToken:
    """
    Public API: token -> LemmaToken (Marathi via Apertium).
    """
    lemma, pos, analyses, is_oov, is_ambiguous = _lemmatize_cached("mar", token)
    return LemmaToken(
        token=token,
        lemma=lemma,
        pos=pos,
        analyses=list(analyses) if analyses else None,
        is_oov=is_oov,
        is_ambiguous=is_ambiguous,
    )
