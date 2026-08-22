"""
Exercises the FastAPI layer's request/response contract without ever
constructing a real RagPipeline (which would need network access to
download the embedding/reranker/NLI models, plus a Groq API key). We
build the FastAPI app directly (bypassing the lifespan startup hook) and
inject a fake pipeline via dependency override.
"""

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_pipeline, router
from fastapi import FastAPI


@dataclass
class _FakeIndex:
    ntotal: int = 5


@dataclass
class _FakeVectorStore:
    index: _FakeIndex = field(default_factory=_FakeIndex)


@dataclass
class _FakeAnswerResult:
    answer: str
    refused: bool
    overall_confidence: float
    coverage_ratio: float
    session_id: str
    claims: list = field(default_factory=list)
    latency_seconds: float = 0.01
    used_llm: bool = True


class _FakePipeline:
    def __init__(self, canned_result: _FakeAnswerResult):
        self.vector_store = _FakeVectorStore()
        self._canned_result = canned_result
        self.last_question = None
        self.last_session_id = None

    def answer(self, question, session_id=None):
        self.last_question = question
        self.last_session_id = session_id
        return self._canned_result


def _build_app(fake_pipeline: _FakePipeline) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    return app


def test_health_endpoint_reports_index_size():
    fake = _FakePipeline(_FakeAnswerResult(answer="", refused=False, overall_confidence=0, coverage_ratio=0, session_id="s"))
    client = TestClient(_build_app(fake))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["index_vectors"] == 5


def test_chat_returns_answer_with_citations():
    canned = _FakeAnswerResult(
        answer="The UE shall support NR-U. [TS 38.331, Clause 5.2.1].",
        refused=False,
        overall_confidence=0.87,
        coverage_ratio=1.0,
        session_id="abc-123",
        claims=[
            type(
                "V",
                (),
                {
                    "claim": "The UE shall support NR-U.",
                    "citation": "TS 38.331, Clause 5.2.1",
                    "confidence": 0.87,
                    "verification_method": "nli",
                    "supported": True,
                    "matched_passage": "The UE shall support NR-U operations in band n46.",
                },
            )()
        ],
    )
    fake = _FakePipeline(canned)
    client = TestClient(_build_app(fake))

    resp = client.post("/api/chat", json={"question": "Does the UE support NR-U?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["session_id"] == "abc-123"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["citation"] == "TS 38.331, Clause 5.2.1"
    assert fake.last_question == "Does the UE support NR-U?"


def test_chat_refusal_has_no_citations():
    canned = _FakeAnswerResult(
        answer="I couldn't find this information in the provided 3GPP specifications.",
        refused=True,
        overall_confidence=0.2,
        coverage_ratio=0.0,
        session_id="xyz",
        claims=[],
    )
    fake = _FakePipeline(canned)
    client = TestClient(_build_app(fake))

    resp = client.post("/api/chat", json={"question": "What is the airspeed of a swallow?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_chat_rejects_empty_question():
    fake = _FakePipeline(_FakeAnswerResult(answer="", refused=False, overall_confidence=0, coverage_ratio=0, session_id="s"))
    client = TestClient(_build_app(fake))
    resp = client.post("/api/chat", json={"question": ""})
    assert resp.status_code == 422


def test_session_reset_endpoint():
    fake = _FakePipeline(_FakeAnswerResult(answer="", refused=False, overall_confidence=0, coverage_ratio=0, session_id="s"))
    client = TestClient(_build_app(fake))
    resp = client.post("/api/sessions/some-id/reset")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "some-id", "reset": True}
