"""
Cross-encoder reranking stage.

Bi-encoder FAISS search (embedder.py) is fast but approximate: it scores
query and passage independently, so it can under- or over-rank passages
that share vocabulary without being truly relevant. This module re-scores
the top-N FAISS candidates with a cross-encoder that reads the query and
passage *together*, and returns a re-ordered top-K.

If the reranker model can't be loaded (offline environment, disabled via
config), we degrade gracefully to the FAISS ordering rather than failing
the whole request — reranking is a quality improvement, not a hard
dependency.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.ingestion.chunker import ChunkMetadata
from app.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    content: str
    metadata: ChunkMetadata
    retrieval_score: float  # original FAISS cosine similarity
    rerank_score: float  # cross-encoder relevance score (or a copy of retrieval_score if rerank was skipped)


class CrossEncoderReranker:
    _instance: "CrossEncoderReranker | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._load_failed = False

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is None and not self._load_failed:
                try:
                    logger.info("Loading reranker model %s ...", self._model_name)
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self._model_name)
                    logger.info("Reranker model loaded.")
                except Exception:
                    logger.exception(
                        "Failed to load reranker model %s — falling back to FAISS ordering.",
                        self._model_name,
                    )
                    self._load_failed = True
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RerankedChunk]:
        """Re-score `candidates` against `query` and return the best `top_k`,
        ordered by cross-encoder relevance (falls back to FAISS order if the
        model is unavailable)."""
        if not candidates:
            return []

        model = self._ensure_loaded()

        if model is None:
            ranked = sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)[:top_k]
            return [
                RerankedChunk(
                    content=c.content,
                    metadata=c.metadata,
                    retrieval_score=c.retrieval_score,
                    rerank_score=c.retrieval_score,  # no independent signal available
                )
                for c in ranked
            ]

        pairs = [(query, c.content) for c in candidates]
        raw_scores = [float(s) for s in model.predict(pairs)]

        scored = list(zip(candidates, raw_scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            RerankedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                retrieval_score=chunk.retrieval_score,
                rerank_score=score,
            )
            for chunk, score in scored[:top_k]
        ]

    @classmethod
    def get_singleton(cls, model_name: str) -> "CrossEncoderReranker":
        if cls._instance is None or cls._instance._model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance
