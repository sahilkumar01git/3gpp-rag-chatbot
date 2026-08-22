"""
Integration test that exercises the full non-ML-dependent pipeline against
the REAL bundled sample corpus PDFs (data/raw_pdfs/sample_corpus/) — i.e.
everything except the actual embedding/reranker/NLI *models*, which are
swapped for the deterministic FakeEmbedder/FakeNLIVerifier so this runs
offline in CI. This is the closest thing to a real smoke test of
"drop PDFs in a folder -> get cited, verified answers" without requiring
network access or a Groq API key.
"""

from pathlib import Path

import pytest

from app.ingestion.build_index import build_all_chunks
from app.ingestion.pdf_parser import discover_pdfs
from app.retrieval.vector_store import VectorStore
from app.verification.confidence import verify_answer
from tests.fakes import FakeEmbedder, FakeNLIVerifier

SAMPLE_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"


pytestmark = pytest.mark.skipif(
    not discover_pdfs(SAMPLE_CORPUS_DIR), reason="Sample corpus not generated — run scripts/generate_sample_corpus.py"
)


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory):
    from app.config import settings

    original_dir = settings.raw_pdf_dir
    settings.raw_pdf_dir = SAMPLE_CORPUS_DIR
    try:
        tmp_dir = tmp_path_factory.mktemp("integration_index")
        store = VectorStore(tmp_dir / "faiss.index", tmp_dir / "chunks.json")
        embedder = FakeEmbedder()
        store.build(build_all_chunks(), embedder, source_fingerprint="integration-test")
        yield store, embedder
    finally:
        settings.raw_pdf_dir = original_dir


def test_multiple_specs_are_represented_in_the_index(indexed_store):
    store, _ = indexed_store
    spec_ids = {c.metadata.spec_id for c in store.chunks}
    assert spec_ids == {"TS 38.331", "TS 23.501", "TS 24.501", "TS 38.321", "TS 33.501"}, spec_ids


def test_query_retrieves_relevant_spec_first(indexed_store):
    store, embedder = indexed_store
    # Note: the FakeEmbedder is a crude bag-of-words hash, not a real semantic
    # model, so we query with vocabulary distinctive to the target passage
    # rather than a natural paraphrase — real retrieval-quality evaluation
    # (with the actual sentence-transformer model) lives in eval/run_eval.py.
    query_vec = embedder.encode(["RRC_IDLE RRC_INACTIVE RRC_CONNECTED states"])[0]
    hits = store.search(query_vec, top_n=5)
    assert hits[0].metadata.spec_id == "TS 38.331"


def test_table_chunk_is_retrievable_and_citable(indexed_store):
    store, embedder = indexed_store
    query_vec = embedder.encode(["MCS index modulation order 16QAM 64QAM"])[0]
    hits = store.search(query_vec, top_n=5)
    top_spec_ids = {h.metadata.spec_id for h in hits[:3]}
    assert "TS 38.321" in top_spec_ids


def test_full_flow_produces_grounded_cited_answer(indexed_store):
    """Simulates what the LLM *should* produce for a well-supported question,
    then runs it through the real verification stage against real retrieved
    passages — end to end minus the actual network call to Groq."""
    store, embedder = indexed_store

    question = "What is K_SEAF derived from?"
    query_vec = embedder.encode([question])[0]
    from app.retrieval.reranker import RerankedChunk

    raw_hits = store.search(query_vec, top_n=5)
    passages = [
        RerankedChunk(content=h.content, metadata=h.metadata, retrieval_score=h.retrieval_score, rerank_score=5.0)
        for h in raw_hits
    ]

    simulated_llm_answer = "K_SEAF is derived from K_AUSF."
    result = verify_answer(simulated_llm_answer, passages, embedder, FakeNLIVerifier())

    assert not result.refused
    assert "TS 33.501" in result.final_answer


def test_hallucinated_answer_is_refused_even_with_real_retrieved_context(indexed_store):
    store, embedder = indexed_store

    question = "What RRC states does a UE support?"
    query_vec = embedder.encode([question])[0]
    from app.retrieval.reranker import RerankedChunk

    raw_hits = store.search(query_vec, top_n=5)
    passages = [
        RerankedChunk(content=h.content, metadata=h.metadata, retrieval_score=h.retrieval_score, rerank_score=5.0)
        for h in raw_hits
    ]

    # Plausible-sounding but fabricated — not present in any real passage.
    hallucinated_answer = "A UE in NR supports a fourth RRC state called RRC_SUSPENDED for satellite handover."
    result = verify_answer(hallucinated_answer, passages, embedder, FakeNLIVerifier())

    assert result.refused
