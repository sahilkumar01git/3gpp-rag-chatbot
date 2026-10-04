"""
NLI (Natural Language Inference) claim verifier — the primary signal used
to decide whether a generated sentence is actually supported by a
retrieved passage, replacing the old `difflib.SequenceMatcher` string
similarity check.

Given a (premise=passage, hypothesis=claim) pair, an NLI cross-encoder
predicts entailment / neutral / contradiction. We use the entailment
probability as the "is this claim actually supported by this evidence?"
score. This catches the failure modes that string similarity cannot:

- A claim can share almost no vocabulary with its supporting passage
  ("the timer must expire before retransmission" vs. "T1 shall lapse
  prior to the retransmit attempt") and still be correctly recognized as
  entailed.
- A claim can share *heavy* vocabulary with a passage while asserting the
  opposite or a different number ("EIRP shall not exceed 20 dBm/MHz" vs.
  "EIRP shall not exceed 23 dBm/MHz") and still be correctly flagged as
  NOT entailed (SequenceMatcher would score these as highly similar).

As with the reranker, if the model can't be loaded we degrade gracefully
rather than crash — callers should treat a `None` model as "NLI signal
unavailable" and fall back to the embedding-similarity verifier instead
(see confidence.py, which blends both).
"""

from __future__ import annotations

import logging
import re
import threading

from app.retrieval.reranker import RerankedChunk

logger = logging.getLogger(__name__)

_PREMISE_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _PREMISE_SENTENCE_RE.split(text) if s.strip()]
    return sentences if len(sentences) > 1 else []


def _table_rows(text: str) -> list[str]:
    """Render a markdown table chunk as one 'Header: value; ...' statement
    per row. An NLI model cannot read a raw pipe table — scored as one blob
    it returns near-random entailment — but it can judge a single row."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    header, body = cells[0], [r for r in cells[1:] if not all(set(c) <= set("-: ") for c in r)]
    return ["; ".join(f"{h}: {v}" for h, v in zip(header, row)) + "." for row in body]


def _premises(text: str) -> list[str]:
    """NLI checkpoints are trained on single-sentence premises. Given a
    whole multi-sentence passage they tend to answer "neutral" even when one
    sentence states the claim outright, so a claim is scored against each
    sentence (or table row) as well, and the best-supported one wins."""
    rows = _table_rows(text)
    if rows:
        return rows
    return [text] + _split_sentences(text)


class NLIVerifier:
    _instance: "NLIVerifier | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._load_failed = False

    def is_available(self) -> bool:
        return self._ensure_loaded() is not None

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is None and not self._load_failed:
                try:
                    logger.info("Loading NLI verification model %s ...", self._model_name)
                    from sentence_transformers import CrossEncoder

                    # cross-encoder/nli-* models output 3 logits: [contradiction, entailment, neutral]
                    self._model = CrossEncoder(self._model_name)
                    logger.info("NLI verification model loaded.")
                except Exception:
                    logger.exception(
                        "Failed to load NLI model %s — falling back to embedding-similarity verification.",
                        self._model_name,
                    )
                    self._load_failed = True
        return self._model

    def entailment_score(self, claim: str, passage: RerankedChunk) -> float | None:
        """Returns P(entailment) in [0, 1], or None if the model is unavailable."""
        model = self._ensure_loaded()
        if model is None:
            return None

        import numpy as np

        premises = _premises(passage.content)
        logits = np.asarray(model.predict([(premise, claim) for premise in premises]))
        # Standard label order for cross-encoder/nli-* checkpoints:
        # [contradiction, entailment, neutral].
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return float(probs[:, 1].max())

    def best_entailment(self, claim: str, passages: list[RerankedChunk]) -> tuple[RerankedChunk | None, float | None]:
        """Score `claim` against every candidate passage and return the
        passage with the highest entailment probability."""
        if not passages:
            return None, None

        best_passage, best_score = None, None
        for passage in passages:
            score = self.entailment_score(claim, passage)
            if score is None:
                return None, None  # model unavailable — let caller fall back
            if best_score is None or score > best_score:
                best_passage, best_score = passage, score
        return best_passage, best_score

    @classmethod
    def get_singleton(cls, model_name: str) -> "NLIVerifier":
        if cls._instance is None or cls._instance._model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance
