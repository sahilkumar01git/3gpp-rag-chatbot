"""
Embedding backend.

`Embedder` is a thin Protocol so tests can swap in a deterministic fake
without downloading any model or touching the network (see
`tests/fakes.py`). `SentenceTransformerEmbedder` is the real production
implementation used everywhere else.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalized embeddings."""
        ...

    @property
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder:
    """Lazy-loaded, process-wide singleton wrapper around a SentenceTransformer
    bi-encoder. Loading is deferred to first use so importing this module
    (e.g. for unit tests of unrelated code) never triggers a model download."""

    _instance: "SentenceTransformerEmbedder | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None  # loaded lazily

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading embedding model %s ...", self._model_name)
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name)
                    logger.info("Embedding model loaded.")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_loaded()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    @property
    def dimension(self) -> int:
        model = self._ensure_loaded()
        return int(model.get_sentence_embedding_dimension())

    @classmethod
    def get_singleton(cls, model_name: str) -> "SentenceTransformerEmbedder":
        if cls._instance is None or cls._instance._model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance
