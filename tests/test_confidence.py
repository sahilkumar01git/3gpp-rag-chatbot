import pytest

from app.generation.prompts import REFUSAL_TEXT
from app.ingestion.chunker import ChunkMetadata
from app.retrieval.reranker import RerankedChunk
from app.verification.confidence import verify_answer
from tests.fakes import FakeEmbedder, FakeNLIVerifier


def _passage(text: str, retrieval_score: float = 0.8, rerank_score: float = 3.0, clause="5.2.1") -> RerankedChunk:
    meta = ChunkMetadata(
        spec_id="TS 38.331",
        doc_type="TS",
        spec_number="38.331",
        version="17.6.0",
        release="17",
        clause_number=clause,
        clause_title="General",
        page_start=10,
        page_end=10,
        source_file="ts_38331.pdf",
    )
    return RerankedChunk(content=text, metadata=meta, retrieval_score=retrieval_score, rerank_score=rerank_score)


def test_well_supported_claim_is_accepted_with_citation():
    passages = [_passage("The UE shall support NR-U operations in band n46 with EIRP up to 20 dBm.")]
    answer = "The UE shall support NR-U operations in band n46 with EIRP up to 20 dBm."
    result = verify_answer(answer, passages, FakeEmbedder(), FakeNLIVerifier())
    assert not result.refused
    assert result.coverage_ratio == 1.0
    assert "TS 38.331" in result.final_answer
    assert result.claims[0].supported


def test_unsupported_claim_triggers_refusal():
    passages = [_passage("The UE shall support TM1 and TM2 RLC modes for data transfer.")]
    # Completely unrelated claim relative to the retrieved evidence.
    answer = "The satellite orbit period is ninety minutes at low earth orbit altitude."
    result = verify_answer(answer, passages, FakeEmbedder(), FakeNLIVerifier())
    assert result.refused
    assert result.final_answer == REFUSAL_TEXT


def test_numeric_mismatch_is_caught_even_with_high_word_overlap():
    """This is the key case difflib.SequenceMatcher gets wrong: the claim and
    the passage share almost every word, but assert two different numbers."""
    passages = [_passage("The maximum allowed EIRP for NR-U shall not be greater than 20 dBm per MHz.")]
    wrong_number_answer = "The maximum allowed EIRP for NR-U shall not be greater than 99 dBm per MHz."
    result = verify_answer(wrong_number_answer, passages, FakeEmbedder(), FakeNLIVerifier())
    assert result.refused, "A claim with a fabricated number should NOT be accepted just because wording matches."


def test_llm_self_refusal_is_passed_through_unchanged():
    result = verify_answer(REFUSAL_TEXT, [_passage("irrelevant")], FakeEmbedder(), FakeNLIVerifier())
    assert result.refused
    assert result.final_answer == REFUSAL_TEXT


def test_no_passages_at_all_forces_refusal():
    result = verify_answer("Some answer.", [], FakeEmbedder(), FakeNLIVerifier())
    assert result.refused


def test_partial_answer_keeps_only_supported_claims():
    passages = [
        _passage("The UE shall support TM1 and TM2 RLC modes for data transfer.", clause="5.3.1"),
    ]
    answer = (
        "The UE shall support TM1 and TM2 RLC modes for data transfer. "
        "The satellite orbit period is ninety minutes at low earth orbit altitude."
    )
    result = verify_answer(answer, passages, FakeEmbedder(), FakeNLIVerifier())
    assert not result.refused
    assert result.coverage_ratio == 0.5
    assert "TM1" in result.final_answer
    assert "satellite orbit" not in result.final_answer


@pytest.mark.parametrize("nli", [None])
def test_falls_back_to_embedding_similarity_when_nli_unavailable(nli):
    passages = [_passage("The UE shall support NR-U operations in band n46.")]
    answer = "The UE shall support NR-U operations in band n46."
    result = verify_answer(answer, passages, FakeEmbedder(), nli)
    assert not result.refused
    assert result.claims[0].verification_method == "embedding_similarity"
