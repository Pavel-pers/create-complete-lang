"""
WordNet utilities for coverage analysis and gap validation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_WN_READY = False


def _ensure_wordnet() -> None:
    global _WN_READY
    if _WN_READY:
        return
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    _WN_READY = True


def wn_coverage(words: list[str]) -> dict:
    """Fraction of words with at least one WordNet synset (lemma match)."""
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    n = len(words)
    has_synset = 0
    synset_counts = []
    for w in words:
        s = wn.synsets(w)
        if s:
            has_synset += 1
            synset_counts.append(len(s))
    return {
        "n_words": n,
        "n_in_wordnet": has_synset,
        "coverage": has_synset / n if n > 0 else 0.0,
        "mean_synsets_per_covered": float(np.mean(synset_counts)) if synset_counts else 0.0,
    }


def wn_supersenses(words: list[str]) -> dict[str, list[str]]:
    """Map each word to its list of lexical-file names (supersenses)."""
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    result: dict[str, list[str]] = {}
    for w in words:
        ss = [s.lexname() for s in wn.synsets(w)]
        if ss:
            result[w] = ss
    return result


def cluster_dominant_supersense(
    words: list[str],
    min_coverage: float = 0.1,
) -> Optional[tuple[str, float]]:
    """Find the most common WN supersense in a cluster.
    Returns (supersense, fraction) or None if no coverage.
    """
    _ensure_wordnet()
    from collections import Counter
    from nltk.corpus import wordnet as wn

    counter: Counter[str] = Counter()
    hits = 0
    for w in words:
        syns = wn.synsets(w)
        if syns:
            hits += 1
            # First synset is most common sense
            counter[syns[0].lexname()] += 1
    if hits / max(len(words), 1) < min_coverage:
        return None
    if not counter:
        return None
    top, cnt = counter.most_common(1)[0]
    return top, cnt / len(words)


def archaic_markers(words: list[str]) -> dict:
    """Heuristic archaism detection.

    1. Word NOT in WordNet at all → likely archaic/dialect/OOV
    2. Word ends in -eth, -est, or contains distinctive EME spellings (ae, oe, y-)
    """
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    EME_ENDINGS = ("eth", "est", "th",)
    EME_PATTERNS = ("haue", "selfe", "euer", "owne", "loue", "thinke", "onely", "euill")

    not_in_wn = []
    archaic_morph = []
    for w in words:
        if not wn.synsets(w):
            not_in_wn.append(w)
        if w in EME_PATTERNS:
            archaic_morph.append(w)
        elif len(w) > 4 and any(w.endswith(e) for e in EME_ENDINGS) and not wn.synsets(w):
            archaic_morph.append(w)
    return {
        "not_in_wn": not_in_wn,
        "archaic_morph": archaic_morph,
        "frac_not_in_wn": len(not_in_wn) / max(len(words), 1),
    }


def has_synset_in_cluster(
    target_word: str,
    cluster_words: list[str],
) -> bool:
    """Check if any synonym (same synset) of target_word exists in cluster_words."""
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    target_synsets = wn.synsets(target_word)
    if not target_synsets:
        return False
    cluster_set = set(cluster_words)
    for syn in target_synsets:
        for lemma in syn.lemma_names():
            if lemma.lower() in cluster_set:
                return True
    return False


def validate_gap_candidate(
    candidate_word: str,
    acceptor_cluster_words: list[str],
) -> dict:
    """A candidate is a "real" gap if its WordNet synsets have no synonym
    in the acceptor cluster.

    Returns dict with:
    - in_wordnet: bool (is the candidate itself in WN?)
    - synonym_in_acceptor: bool
    - is_gap: bool (in WN AND no synonym in acceptor)
    """
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    syns = wn.synsets(candidate_word)
    if not syns:
        return {"in_wordnet": False, "synonym_in_acceptor": False, "is_gap": False}
    acceptor_set = {w.lower() for w in acceptor_cluster_words}
    syn_in_acceptor = False
    for syn in syns:
        for lemma in syn.lemma_names():
            if lemma.lower() != candidate_word.lower() and lemma.lower() in acceptor_set:
                syn_in_acceptor = True
                break
        if syn_in_acceptor:
            break
    return {
        "in_wordnet": True,
        "synonym_in_acceptor": syn_in_acceptor,
        "is_gap": not syn_in_acceptor,
    }
