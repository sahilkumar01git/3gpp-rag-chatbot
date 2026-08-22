"""
Splits a generated answer into individual factual claims to verify
independently. Kept separate from the verifiers themselves so the
splitting heuristic can be improved without touching NLI/embedding code.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

MIN_CLAIM_CHARS = 8


def split_into_claims(answer: str) -> list[str]:
    """Best-effort sentence splitter. Not perfect (e.g. abbreviations with
    periods), but claims are re-verified against evidence regardless, so an
    over- or under-split sentence only affects granularity, not correctness."""
    stripped = answer.strip()
    if not stripped:
        return []

    raw_sentences = _SENTENCE_SPLIT_RE.split(stripped)
    claims = [s.strip() for s in raw_sentences if len(s.strip()) >= MIN_CLAIM_CHARS]
    return claims if claims else [stripped]
