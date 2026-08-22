"""
Deterministic, network-free test doubles.

The real embedder/reranker/NLI verifier download models from Hugging Face
on first use, which is neither available nor desirable in unit tests.
These fakes implement the same interfaces with cheap, deterministic
bag-of-words math so retrieval/verification *logic* (ranking, thresholds,
confidence blending) can be tested without any network access or GPU.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

FAKE_DIM = 64


def _hash_bow_vector(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    """A crude but deterministic 'embedding': hash each word into a bucket
    and accumulate. Cosine similarity between two such vectors roughly
    tracks word overlap — good enough to exercise ranking/threshold logic
    in tests without a real model."""
    vec = np.zeros(dim, dtype="float32")
    words = re.findall(r"[a-z0-9]+", text.lower())
    for w in words:
        bucket = int(hashlib.sha256(w.encode()).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class FakeEmbedder:
    """Implements the same `Embedder` protocol as SentenceTransformerEmbedder."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([_hash_bow_vector(t) for t in texts])

    @property
    def dimension(self) -> int:
        return FAKE_DIM


class FakeNLIVerifier:
    """Mimics NLIVerifier's public surface using word-overlap as a stand-in
    for entailment probability, with a simple negation/number-mismatch
    penalty so tests can assert it behaves *unlike* naive string similarity."""

    def is_available(self) -> bool:
        return True

    def entailment_score(self, claim: str, passage) -> float:
        claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
        passage_words = set(re.findall(r"[a-z0-9]+", passage.content.lower()))
        if not claim_words:
            return 0.0
        overlap = len(claim_words & passage_words) / len(claim_words)

        claim_numbers = set(re.findall(r"\d+", claim))
        passage_numbers = set(re.findall(r"\d+", passage.content))
        if claim_numbers and passage_numbers and claim_numbers != passage_numbers:
            overlap *= 0.2  # numeric mismatch -> strongly penalize, unlike SequenceMatcher

        return float(min(overlap, 1.0))

    def best_entailment(self, claim: str, passages):
        if not passages:
            return None, None
        scored = [(p, self.entailment_score(claim, p)) for p in passages]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[0]
