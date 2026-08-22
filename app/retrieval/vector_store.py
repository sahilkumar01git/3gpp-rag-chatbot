"""
FAISS-backed vector store.

Fixes the original bug where the index was rebuilt from scratch on every
process start: `load_or_build()` loads the persisted index + metadata from
disk whenever they exist and are up to date, and only (re)builds when the
index is missing, corrupt, or the caller explicitly asks for a rebuild
(`--rebuild` in the CLI / build script).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.ingestion.chunker import Chunk, ChunkMetadata
from app.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"


@dataclass
class RetrievedChunk:
    content: str
    metadata: ChunkMetadata
    retrieval_score: float  # raw FAISS cosine similarity, in [-1, 1]


class VectorStore:
    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    # ── persistence ──────────────────────────────────────────────────────

    def _manifest_path(self) -> Path:
        return self.index_path.parent / _MANIFEST_NAME

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def is_stale(self, source_fingerprint: str) -> bool:
        """True if the persisted index was built from different source
        content than `source_fingerprint` (e.g. the PDFs changed), so it
        needs rebuilding rather than blindly reused."""
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return True
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return True
        return manifest.get("source_fingerprint") != source_fingerprint

    def load(self) -> None:
        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))
        raw_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.chunks = [
            Chunk(content=item["content"], metadata=ChunkMetadata.from_dict(item["metadata"]))
            for item in raw_meta
        ]
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs {len(self.chunks)} chunk records. "
                "Rebuild the index with `python -m app.ingestion.build_index --rebuild`."
            )
        logger.info("Loaded index with %d vectors.", self.index.ntotal)

    def build(self, chunks: list[Chunk], embedder: Embedder, source_fingerprint: str) -> None:
        if not chunks:
            raise RuntimeError("No chunks were extracted from the source documents — nothing to index.")

        logger.info("Embedding %d chunks ...", len(chunks))
        vectors = embedder.encode([c.content for c in chunks])
        vectors = np.asarray(vectors, dtype="float32")

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                [{"content": c.content, "metadata": c.metadata.to_dict()} for c in chunks],
                f,
                indent=2,
                ensure_ascii=False,
            )
        self._manifest_path().write_text(
            json.dumps({"source_fingerprint": source_fingerprint, "num_chunks": len(chunks)}, indent=2)
        )

        self.index = index
        self.chunks = chunks
        logger.info("Built and persisted FAISS index with %d vectors at %s", index.ntotal, self.index_path)

    def load_or_build(
        self,
        chunks_provider,
        embedder: Embedder,
        source_fingerprint: str,
        force_rebuild: bool = False,
    ) -> None:
        """`chunks_provider` is a zero-arg callable returning `list[Chunk]`,
        only invoked if we actually need to (re)build — so callers don't pay
        the cost of re-parsing PDFs when a fresh index is already on disk."""
        if not force_rebuild and self.exists() and not self.is_stale(source_fingerprint):
            self.load()
            return

        if not force_rebuild and self.exists():
            logger.info("Persisted index is stale relative to source documents — rebuilding.")
        chunks = chunks_provider()
        self.build(chunks, embedder, source_fingerprint)

    # ── search ───────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_n: int) -> list[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("Vector store not loaded — call load_or_build() first.")

        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(top_n, self.index.ntotal))

        hits: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            hits.append(RetrievedChunk(content=chunk.content, metadata=chunk.metadata, retrieval_score=float(score)))
        return hits


def fingerprint_pdf_dir(raw_pdf_dir: Path) -> str:
    """Cheap content fingerprint (path + mtime + size of every PDF) used to
    decide whether a persisted index is stale, without re-parsing anything."""
    h = hashlib.sha256()
    for pdf_path in sorted(raw_pdf_dir.rglob("*.pdf")):
        stat = pdf_path.stat()
        h.update(f"{pdf_path}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return h.hexdigest()
