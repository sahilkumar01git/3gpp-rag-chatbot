"""
Embedding-similarity claim verifier.

This is the "always available" verification signal: it reuses the same
bi-encoder already loaded for retrieval (no extra model download) and
scores each claim against each candidate passage by cosine similarity.
It's strictly better than the original `difflib.SequenceMatcher` approach
because it compares *meaning* rather than surface character overlap — two
sentences that say the same thing in different words will score highly
here, while SequenceMatcher would score them poorly; conversely two
sentences that share vocabulary but assert different things (e.g. differ
by a negation or a swapped number) tend to score lower here than the
inflated overlap score SequenceMatcher would give them.

It is intentionally treated as a *fallback/floor* signal, not the primary
one — see verification/nli_verifier.py and confidence.py for the primary,
entailment-based signal used when that model is available.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.embedder import Embedder
from app.retrieval.reranker import RerankedChunk


def best_match_by_embedding(
    claim: str, passages: list[RerankedChunk], embedder: Embedder
) -> tuple[RerankedChunk | None, float]:
    """Return the passage most semantically similar to `claim`, and the
    cosine similarity score in [-1, 1] (typically [0, 1] in practice)."""
    if not passages:
        return None, 0.0

    vectors = embedder.encode([claim] + [p.content for p in passages])
    claim_vec, passage_vecs = vectors[0], vectors[1:]

    sims = passage_vecs @ claim_vec  # already L2-normalized -> cosine similarity
    best_idx = int(np.argmax(sims))
    return passages[best_idx], float(sims[best_idx])
