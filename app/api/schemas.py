from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Omit to start a new conversation.")


class CitationOut(BaseModel):
    claim: str
    citation: str | None
    confidence: float
    verification_method: str
    evidence_snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    session_id: str
    overall_confidence: float
    coverage_ratio: float
    citations: list[CitationOut]
    latency_seconds: float
    used_llm: bool


class HealthResponse(BaseModel):
    status: str
    index_vectors: int
    reranker_enabled: bool
    nli_verifier_enabled: bool
    groq_configured: bool


class ResetSessionResponse(BaseModel):
    session_id: str
    reset: bool
