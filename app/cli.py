"""
Command-line chat interface — kept as a lightweight alternative to the
FastAPI + web UI, but now sharing the exact same `RagPipeline` (retrieval
-> rerank -> generate -> verify) instead of its own copy of the logic.

    python -m app.cli
    python -m app.cli --rebuild-index
"""

from __future__ import annotations

import argparse
import logging

from app.generation.llm_client import LLMError
from app.logging_config import configure_logging
from app.pipeline import RagPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="3GPP RAG chatbot (CLI)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force a full index rebuild before starting.")
    args = parser.parse_args()

    print("Loading index and models ...")
    pipeline = RagPipeline(force_rebuild=args.rebuild_index)
    print(f"Ready — index has {pipeline.vector_store.index.ntotal} chunks.")
    print("Type 'exit' to quit, 'reset' to clear conversation memory.\n")

    session_id: str | None = None

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "reset":
            session_id = None
            print("Conversation memory cleared.\n")
            continue
        if not question:
            continue

        try:
            result = pipeline.answer(question, session_id=session_id)
        except LLMError as exc:
            print(f"\nError talking to the LLM: {exc}\n")
            continue

        session_id = result.session_id

        print()
        print("Answer:", result.answer)
        print(
            f"[confidence={result.overall_confidence:.2f} "
            f"coverage={result.coverage_ratio:.2f} "
            f"latency={result.latency_seconds:.2f}s]"
        )
        if not result.refused:
            unsupported = [c for c in result.claims if not c.supported]
            if unsupported:
                print(f"({len(unsupported)} claim(s) from the draft answer were dropped as unverified.)")
        print()


if __name__ == "__main__":
    main()
