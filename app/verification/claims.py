"""
Splits a generated answer into individual factual claims to verify
independently. Kept separate from the verifiers themselves so the
splitting heuristic can be improved without touching NLI/embedding code.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

MIN_CLAIM_CHARS = 8

# Formatting the LLM adds around a fact: citation markers ([1], 【1】) and
# markdown emphasis. None of it is part of the claim, and it measurably
# hurts NLI entailment scoring, so it is stripped before verification.
_CITATION_MARKER_RE = re.compile(r"\s*(?:\[\d+\]|【[^】]*】)")
_MARKDOWN_EMPHASIS_RE = re.compile(r"\*\*|__|`")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_UNICODE_HYPHENS = str.maketrans({"‐": "-", "‑": "-"})


def clean_claim(text: str) -> str:
    text = _CITATION_MARKER_RE.sub("", text)
    text = _MARKDOWN_EMPHASIS_RE.sub("", text)
    text = text.translate(_UNICODE_HYPHENS)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return " ".join(text.split())


def split_into_claims(answer: str) -> list[str]:
    """Best-effort sentence splitter. Not perfect (e.g. abbreviations with
    periods), but claims are re-verified against evidence regardless, so an
    over- or under-split sentence only affects granularity, not correctness."""
    stripped = answer.strip()
    if not stripped:
        return []

    raw_sentences = [clean_claim(s) for s in _SENTENCE_SPLIT_RE.split(stripped)]
    claims = [s for s in raw_sentences if len(s) >= MIN_CLAIM_CHARS]
    return claims if claims else [clean_claim(stripped) or stripped]
