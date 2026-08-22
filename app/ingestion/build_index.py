"""
Standalone ingestion entry point.

    python -m app.ingestion.build_index            # build only if missing/stale
    python -m app.ingestion.build_index --rebuild   # force a full rebuild

This is also called internally by the API/CLI startup path
(app/pipeline.py) via `load_or_build`, so in normal operation you never
need to run it by hand — it exists as an explicit, scriptable step for
CI, Docker image builds, or when you've just dropped new PDFs into
data/raw_pdfs/ and want to see ingestion logs without starting the server.
"""

from __future__ import annotations

import argparse
import logging

from app.config import settings
from app.ingestion.chunker import Chunk, attach_tables_as_chunks, chunk_document
from app.ingestion.pdf_parser import discover_pdfs, parse_pdf
from app.logging_config import configure_logging
from app.retrieval.embedder import SentenceTransformerEmbedder
from app.retrieval.vector_store import VectorStore, fingerprint_pdf_dir

logger = logging.getLogger(__name__)


def build_all_chunks() -> list[Chunk]:
    """Parse every PDF under `settings.raw_pdf_dir` and return all chunks
    (clause-text chunks + table chunks) across all documents."""
    pdf_paths = discover_pdfs(settings.raw_pdf_dir)
    if not pdf_paths:
        raise RuntimeError(
            f"No PDFs found under {settings.raw_pdf_dir}. "
            "Add real 3GPP spec PDFs there, or run "
            "`python scripts/generate_sample_corpus.py` to generate the bundled "
            "synthetic demo corpus."
        )

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        doc = parse_pdf(pdf_path)
        all_chunks.extend(chunk_document(doc))
        all_chunks.extend(attach_tables_as_chunks(doc))

    logger.info("Parsed %d PDF(s) into %d total chunks.", len(pdf_paths), len(all_chunks))
    return all_chunks


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build/refresh the 3GPP FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Force a full rebuild even if an index already exists.")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder.get_singleton(settings.embed_model_name)
    store = VectorStore(settings.index_path, settings.meta_path)
    fingerprint = fingerprint_pdf_dir(settings.raw_pdf_dir)
    fingerprint = f"{fingerprint}:{settings.embed_model_name}"  # see app/pipeline.py for why

    store.load_or_build(build_all_chunks, embedder, fingerprint, force_rebuild=args.rebuild)
    logger.info("Index ready with %d vectors at %s", store.index.ntotal, settings.index_path)


if __name__ == "__main__":
    main()
