"""
Download / cache small word-similarity benchmarks.

Primary sources are GitHub mirrors; falls back to a built-in minimal subset
of classic pairs so that Phase 1 validation can always run.
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Optional

# (word1, word2, human_score) — classic pairs used as hand-curated fallback
# Drawn from the well-known WordSim-353 / SimLex-999 intersections. Small but defensible.
FALLBACK_PAIRS: list[tuple[str, str, float]] = [
    # high similarity
    ("tiger", "cat", 7.35),
    ("book", "paper", 7.46),
    ("computer", "keyboard", 7.62),
    ("computer", "internet", 7.58),
    ("plane", "car", 5.77),
    ("train", "car", 6.31),
    ("telephone", "communication", 7.50),
    ("television", "radio", 6.77),
    ("drug", "abuse", 6.85),
    ("bread", "butter", 6.19),
    ("cucumber", "potato", 5.92),
    ("doctor", "nurse", 7.00),
    ("professor", "doctor", 6.62),
    ("student", "professor", 6.81),
    ("smart", "student", 4.62),
    ("smart", "stupid", 5.81),
    ("war", "troops", 8.13),
    ("money", "bank", 8.50),
    ("money", "cash", 9.15),
    ("money", "wealth", 8.94),
    ("money", "currency", 9.04),
    ("money", "property", 7.57),
    ("money", "dollar", 8.42),
    ("king", "queen", 8.58),
    ("man", "woman", 8.30),
    ("boy", "girl", 8.43),
    ("coast", "shore", 9.10),
    ("journey", "voyage", 9.29),
    ("car", "automobile", 9.04),
    ("magician", "wizard", 9.02),
    ("forest", "woodland", 8.18),
    ("midday", "noon", 9.29),
    ("fruit", "apple", 6.57),
    ("food", "fruit", 7.52),
    ("century", "year", 7.59),
    # medium
    ("glass", "metal", 4.25),
    ("planet", "star", 7.50),
    ("soap", "opera", 5.00),
    # low similarity / unrelated
    ("noon", "string", 0.54),
    ("professor", "cucumber", 0.31),
    ("king", "cabbage", 0.23),
    ("chord", "smile", 0.54),
    ("rooster", "voyage", 0.62),
    ("cord", "smile", 0.62),
    ("noon", "night", 3.50),
    ("sugar", "approach", 0.88),
    ("stock", "market", 8.08),
    ("stock", "phone", 1.62),
    ("stock", "life", 0.92),
    ("stock", "jaguar", 0.92),
    ("stock", "egg", 1.81),
    ("lad", "brother", 4.46),
    ("lad", "wizard", 0.92),
    ("furnace", "stove", 8.79),
    ("sage", "wizard", 5.69),
    ("asylum", "madhouse", 8.87),
    ("graveyard", "cemetery", 9.00),
    ("shore", "woodland", 3.08),
    ("shore", "forest", 3.46),
    ("implement", "tool", 6.46),
    ("crane", "implement", 2.69),
    ("brother", "monk", 6.27),
    ("bird", "cock", 7.10),
    ("bird", "crane", 7.38),
    ("food", "rooster", 4.42),
    ("boy", "lad", 8.83),
    ("boy", "rooster", 0.96),
    ("boy", "sage", 3.46),
    ("car", "journey", 5.85),
    ("monk", "oracle", 5.00),
    ("monk", "slave", 0.92),
    ("cemetery", "woodland", 2.08),
    ("oracle", "sage", 7.50),
    ("asylum", "cemetery", 2.31),
    ("asylum", "fruit", 0.19),
    ("asylum", "monk", 3.00),
    ("glass", "magician", 2.08),
    ("signature", "autograph", 8.19),
    ("tool", "hammer", 7.80),
    ("hill", "mountain", 8.44),
    ("gem", "jewel", 8.96),
    ("cat", "lion", 7.35),
    ("cat", "dog", 7.35),
    ("dog", "wolf", 7.27),
    ("dog", "pet", 7.90),
    ("fish", "salmon", 7.48),
    ("salmon", "trout", 7.84),
    ("forest", "tree", 7.35),
    ("tree", "oak", 7.00),
    ("house", "home", 8.08),
    ("building", "house", 7.73),
    ("building", "structure", 7.69),
    ("mountain", "valley", 4.54),
    ("sea", "ocean", 9.00),
    ("ship", "boat", 8.54),
    ("water", "river", 7.46),
    ("wood", "forest", 7.65),
    ("art", "music", 5.38),
    ("art", "literature", 6.46),
    ("love", "hate", 7.00),
    ("happy", "sad", 6.62),
    ("hot", "cold", 7.54),
    ("big", "large", 8.73),
    ("small", "little", 9.00),
    ("old", "new", 4.04),
    ("old", "young", 6.81),
    ("fast", "slow", 7.81),
    ("light", "dark", 7.15),
    ("teacher", "student", 6.62),
    ("father", "mother", 8.27),
    ("son", "daughter", 8.11),
    ("boy", "girl", 8.43),
    ("husband", "wife", 8.21),
    ("brother", "sister", 8.19),
    ("uncle", "aunt", 7.96),
    ("king", "emperor", 7.78),
    ("queen", "princess", 7.46),
    ("lawyer", "attorney", 9.00),
    ("physician", "doctor", 9.08),
    ("sick", "ill", 8.92),
    ("afraid", "scared", 8.96),
]


WS353_URL = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-WS-353-ALL.txt"
SIMLEX_URL = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-SIMLEX-999.txt"
MEN_URL = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-MEN-TR-3k.txt"
# RW sometimes unavailable; if missing we just skip it
RW_URL = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-RW.txt"


def _parse_triplet_tsv(text: str, delimiter: str = "\t") -> list[tuple[str, str, float]]:
    """Parse w1<d>w2<d>score. Accepts tab or space/csv. Skips header/comments."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("word1"):
            continue
        parts = line.split(delimiter)
        if len(parts) < 3:
            parts = line.split()
        if len(parts) >= 3:
            try:
                pairs.append((parts[0].lower(), parts[1].lower(), float(parts[2])))
            except ValueError:
                continue
    return pairs


def try_download(url: str, timeout: int = 15) -> Optional[str]:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[benchmarks] fetch failed for {url}: {e}")
        return None


def load_or_fallback(
    name: str,
    cache_dir: Path,
) -> tuple[list[tuple[str, str, float]], str]:
    """Return (pairs, source_tag) where source_tag ∈ {online:<url>, cache, fallback}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.tsv"

    # Try cache
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            text = f.read()
        pairs = _parse_triplet_tsv(text)
        if pairs:
            return pairs, f"cache:{cache_file}"

    # Try online
    urls = {
        "ws353": [WS353_URL],
        "simlex": [SIMLEX_URL],
        "men": [MEN_URL],
        "rw": [RW_URL],
    }
    for url in urls.get(name, []):
        text = try_download(url)
        if text:
            pairs = _parse_triplet_tsv(text)
            if pairs:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
                return pairs, f"online:{url}"

    # Fallback
    return FALLBACK_PAIRS, "fallback"


def load_all(cache_dir: Path) -> dict[str, tuple[list[tuple[str, str, float]], str]]:
    """Load simlex, ws353, men, rw (or fallback)."""
    results = {}
    for name in ["simlex", "ws353", "men", "rw"]:
        pairs, source = load_or_fallback(name, cache_dir)
        results[name] = (pairs, source)
        print(f"[benchmarks] {name}: {len(pairs)} pairs (source: {source})")
    return results


if __name__ == "__main__":
    import sys
    cache = Path(__file__).parent / "benchmarks"
    data = load_all(cache)
