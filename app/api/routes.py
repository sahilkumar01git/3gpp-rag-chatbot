from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    HealthResponse,
    ResetSessionResponse,
)
from app.config import settings
from app.generation.llm_client import LLMConfigError, LLMError
from app.memory.conversation import conversation_store
from app.pipeline import RagPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request) -> RagPipeline:
    """The pipeline (loaded FAISS index + models) is built once at app
    startup and stored on `app.state` — see app/api/main.py — so every
    request reuses it instead of rebuilding anything."""
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse)
def health(pipeline: RagPipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        index_vectors=pipeline.vector_store.index.ntotal if pipeline.vector_store.index else 0,
        reranker_enabled=settings.enable_reranker,
        nli_verifier_enabled=settings.enable_nli_verifier,
        groq_configured=bool(settings.groq_api_key),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> ChatResponse:
    logger.info("Chat request received (session_id=%s)", payload.session_id, extra={"session_id": payload.session_id or "-"})

    try:
        result = pipeline.answer(payload.question, session_id=payload.session_id)
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    citations = [
        CitationOut(
            claim=c.claim,
            citation=c.citation,
            confidence=c.confidence,
            verification_method=c.verification_method,
            evidence_snippet=c.matched_passage,
        )
        for c in result.claims
        if c.supported
    ]

    return ChatResponse(
        answer=result.answer,
        refused=result.refused,
        session_id=result.session_id,
        overall_confidence=result.overall_confidence,
        coverage_ratio=result.coverage_ratio,
        citations=citations,
        latency_seconds=round(result.latency_seconds, 3),
        used_llm=result.used_llm,
    )


@router.post("/sessions/new")
def new_session() -> dict:
    return {"session_id": conversation_store.create_session()}


@router.post("/sessions/{session_id}/reset", response_model=ResetSessionResponse)
def reset_session(session_id: str) -> ResetSessionResponse:
    conversation_store.reset(session_id)
    return ResetSessionResponse(session_id=session_id, reset=True)
