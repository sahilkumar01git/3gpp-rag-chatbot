"""
Ties together retrieval, reranking, and entailment signals into a single
per-claim confidence score, and decides whether the chatbot should answer
or refuse.

This directly addresses two of the weakest points of the original
implementation:

  - "Hallucination detection is weak — SequenceMatcher is not reliable."
    -> Each claim is now checked with an NLI entailment model first
       (nli_verifier.py), falling back to semantic embedding similarity
       (embedding_verifier.py) only if the NLI model is unavailable.
       Both are meaning-based, not character-overlap-based.

  - "No proper confidence/evidence threshold to decide when to refuse."
    -> `MIN_CONFIDENCE_SCORE` and `MIN_RETRIEVAL_SCORE` (app/config.py)
       are combined into one composite score per claim, and the bot
       refuses (falls back to REFUSAL_TEXT) whenever no claim in the
       answer clears both bars.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.config import settings
from app.generation.prompts import REFUSAL_TEXT
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import RerankedChunk
from app.verification.claims import split_into_claims
from app.verification.embedding_verifier import best_match_by_embedding
from app.verification.nli_verifier import NLIVerifier

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    confidence: float
    verification_method: str  # "nli" | "embedding_similarity" | "no_evidence"
    citation: str | None = None
    matched_passage: str | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    semantic_score: float | None = None  # entailment prob, or embedding cosine sim if NLI unavailable


@dataclass
class VerificationResult:
    final_answer: str
    refused: bool
    overall_confidence: float
    coverage_ratio: float  # fraction of extracted claims that were supported
    claims: list[ClaimVerdict] = field(default_factory=list)


def _score_claim(
    claim: str,
    passages: list[RerankedChunk],
    embedder: Embedder,
    nli_verifier: NLIVerifier | None,
) -> ClaimVerdict:
    if not passages:
        return ClaimVerdict(claim=claim, supported=False, confidence=0.0, verification_method="no_evidence")

    semantic_score: float | None = None
    method = "embedding_similarity"
    matched_passage: RerankedChunk | None = None

    if nli_verifier is not None and settings.enable_nli_verifier:
        matched_passage, semantic_score = nli_verifier.best_entailment(claim, passages)
        if semantic_score is not None:
            method = "nli"

    if semantic_score is None:
        matched_passage, semantic_score = best_match_by_embedding(claim, passages, embedder)
        method = "embedding_similarity"

    if matched_passage is None or semantic_score is None:
        return ClaimVerdict(claim=claim, supported=False, confidence=0.0, verification_method="no_evidence")

    retrieval_component = _clip01(matched_passage.retrieval_score)
    rerank_component = _sigmoid(matched_passage.rerank_score)
    semantic_component = _clip01(semantic_score)

    composite = (
        settings.weight_retrieval * retrieval_component
        + settings.weight_rerank * rerank_component
        + settings.weight_entailment * semantic_component
    )

    # Semantic entailment is a hard gate, not just one term in the weighted
    # sum: a claim that is topically close to a passage (high retrieval/
    # rerank scores) but not actually *entailed* by it — e.g. it asserts a
    # different number, or contradicts the passage — must not be rescued by
    # those other two signals. Both conditions must hold independently.
    passes_semantic_gate = semantic_component >= settings.min_semantic_score
    passes_composite_bar = composite >= settings.min_confidence_score
    passes_retrieval_floor = matched_passage.retrieval_score >= settings.min_retrieval_score

    supported = passes_semantic_gate and passes_composite_bar and passes_retrieval_floor

    return ClaimVerdict(
        claim=claim,
        supported=supported,
        confidence=round(composite, 4),
        verification_method=method,
        citation=matched_passage.metadata.citation() if supported else None,
        matched_passage=matched_passage.content if supported else None,
        retrieval_score=round(matched_passage.retrieval_score, 4),
        rerank_score=round(matched_passage.rerank_score, 4),
        semantic_score=round(semantic_score, 4),
    )


def verify_answer(
    raw_answer: str,
    passages: list[RerankedChunk],
    embedder: Embedder,
    nli_verifier: NLIVerifier | None = None,
) -> VerificationResult:
    """Main entry point used by the RAG pipeline (app/api/routes.py and
    app/cli.py) right after the LLM produces `raw_answer`."""

    if raw_answer.strip() == REFUSAL_TEXT or not passages:
        return VerificationResult(
            final_answer=REFUSAL_TEXT,
            refused=True,
            overall_confidence=1.0,  # confident that refusing is the right call here
            coverage_ratio=0.0,
            claims=[],
        )

    claims = split_into_claims(raw_answer)
    verdicts = [_score_claim(claim, passages, embedder, nli_verifier) for claim in claims]

    supported = [v for v in verdicts if v.supported]
    coverage_ratio = len(supported) / len(verdicts) if verdicts else 0.0

    if not supported:
        best_confidence = max((v.confidence for v in verdicts), default=0.0)
        logger.info(
            "Refusing to answer — no claim cleared the confidence threshold "
            "(best claim confidence=%.3f, required>=%.3f).",
            best_confidence,
            settings.min_confidence_score,
        )
        return VerificationResult(
            final_answer=REFUSAL_TEXT,
            refused=True,
            overall_confidence=best_confidence,
            coverage_ratio=0.0,
            claims=verdicts,
        )

    answer_text = " ".join(f"{v.claim} [{v.citation}]." for v in supported)
    overall_confidence = sum(v.confidence for v in supported) / len(supported)

    return VerificationResult(
        final_answer=answer_text,
        refused=False,
        overall_confidence=round(overall_confidence, 4),
        coverage_ratio=round(coverage_ratio, 4),
        claims=verdicts,
    )
