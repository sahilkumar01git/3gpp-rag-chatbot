"""
The single RAG pipeline shared by the FastAPI backend (app/api/routes.py)
and the CLI (app/cli.py) — this is what previously lived, duplicated and
tangled with I/O, directly inside `main()` in the original script.

    retrieve (FAISS) -> rerank (cross-encoder) -> generate (Groq)
        -> verify (NLI/embedding entailment + confidence threshold)
        -> record turn in conversation memory
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import settings
from app.generation.llm_client import LLMError, generate_answer
from app.generation.prompts import build_prompt
from app.ingestion.build_index import build_all_chunks
from app.memory.conversation import conversation_store
from app.retrieval.embedder import Embedder, SentenceTransformerEmbedder
from app.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from app.retrieval.vector_store import RetrievedChunk, VectorStore, fingerprint_pdf_dir
from app.verification.confidence import ClaimVerdict, verify_answer
from app.verification.nli_verifier import NLIVerifier

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    refused: bool
    overall_confidence: float
    coverage_ratio: float
    session_id: str
    retrieved_passages: list[RerankedChunk] = field(default_factory=list)
    claims: list[ClaimVerdict] = field(default_factory=list)
    latency_seconds: float = 0.0
    used_llm: bool = True


class RagPipeline:
    """Owns the loaded index + models for the process lifetime. Construct
    once (see app/api/main.py / app/cli.py) and reuse across requests —
    this is what fixes 'FAISS index rebuilt on every start'."""

    def __init__(self, embedder: Embedder | None = None, force_rebuild: bool = False):
        self.embedder = embedder or SentenceTransformerEmbedder.get_singleton(settings.embed_model_name)
        self.reranker = CrossEncoderReranker.get_singleton(settings.reranker_model_name) if settings.enable_reranker else None
        self.nli_verifier = NLIVerifier.get_singleton(settings.nli_model_name) if settings.enable_nli_verifier else None
        self.vector_store = VectorStore(settings.index_path, settings.meta_path)

        fingerprint = fingerprint_pdf_dir(settings.raw_pdf_dir)
        # Include the embedding model name in the fingerprint: if someone
        # switches EMBED_MODEL_NAME, the persisted index (built with the old
        # model's vector space) must be treated as stale even though the
        # source PDFs haven't changed — otherwise `load_or_build` would
        # happily load an index full of vectors from a different embedding
        # space, and FAISS search against it would be silently wrong (or
        # error out on a dimension mismatch).
        fingerprint = f"{fingerprint}:{settings.embed_model_name}"
        self.vector_store.load_or_build(build_all_chunks, self.embedder, fingerprint, force_rebuild=force_rebuild)

    def _retrieve_and_rerank(self, question: str) -> list[RerankedChunk]:
        query_vector = self.embedder.encode([question])[0]
        candidates: list[RetrievedChunk] = self.vector_store.search(query_vector, settings.retrieval_top_n)

        if not candidates:
            return []

        if self.reranker is not None:
            return self.reranker.rerank(question, candidates, settings.retrieval_top_k)

        # Reranker disabled/unavailable — fall back to plain FAISS ordering,
        # wrapped in the same RerankedChunk shape so downstream code is uniform.
        top = sorted(candidates, key=lambda c: c.retrieval_score, reverse=True)[: settings.retrieval_top_k]
        return [
            RerankedChunk(
                content=c.content,
                metadata=c.metadata,
                retrieval_score=c.retrieval_score,
                rerank_score=c.retrieval_score,
            )
            for c in top
        ]

    def answer(self, question: str, session_id: str | None = None) -> AnswerResult:
        start = time.monotonic()
        question = question.strip()

        passages = self._retrieve_and_rerank(question)

        best_retrieval_score = max((p.retrieval_score for p in passages), default=-1.0)
        if not passages or best_retrieval_score < settings.min_retrieval_score:
            # Nothing relevant enough was even retrieved — refuse before
            # spending an LLM call on it (cheaper, faster, and avoids
            # tempting the model to improvise around thin context).
            logger.info(
                "No passage cleared MIN_RETRIEVAL_SCORE=%.2f (best=%.3f) — refusing without calling the LLM.",
                settings.min_retrieval_score,
                best_retrieval_score,
            )
            new_session_id = conversation_store.append_turn(
                session_id, question, "I couldn't find this information in the provided 3GPP specifications."
            )
            return AnswerResult(
                answer="I couldn't find this information in the provided 3GPP specifications.",
                refused=True,
                overall_confidence=max(best_retrieval_score, 0.0),
                coverage_ratio=0.0,
                session_id=new_session_id,
                retrieved_passages=passages,
                claims=[],
                latency_seconds=time.monotonic() - start,
                used_llm=False,
            )

        history = conversation_store.get_history(session_id)
        prompt = build_prompt(question, passages, history, settings.max_history_turns)

        try:
            raw_answer = generate_answer(prompt)
        except LLMError:
            logger.exception("LLM generation failed for question: %r", question)
            raise

        verification = verify_answer(raw_answer, passages, self.embedder, self.nli_verifier)
        new_session_id = conversation_store.append_turn(session_id, question, verification.final_answer)

        return AnswerResult(
            answer=verification.final_answer,
            refused=verification.refused,
            overall_confidence=verification.overall_confidence,
            coverage_ratio=verification.coverage_ratio,
            session_id=new_session_id,
            retrieved_passages=passages,
            claims=verification.claims,
            latency_seconds=time.monotonic() - start,
            used_llm=True,
        )
