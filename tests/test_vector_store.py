from pathlib import Path

from app.ingestion.chunker import Chunk, ChunkMetadata
from app.retrieval.vector_store import VectorStore, fingerprint_pdf_dir
from tests.fakes import FakeEmbedder


def _chunk(text: str, clause: str) -> Chunk:
    meta = ChunkMetadata(
        spec_id="TS 38.331",
        doc_type="TS",
        spec_number="38.331",
        version="17.6.0",
        release="17",
        clause_number=clause,
        clause_title="Test",
        page_start=1,
        page_end=1,
        source_file="ts_38331.pdf",
    )
    return Chunk(content=text, metadata=meta)


def test_build_then_search_returns_most_similar_chunk_first(tmp_path):
    store = VectorStore(tmp_path / "index" / "faiss.index", tmp_path / "meta" / "chunks.json")
    chunks = [
        _chunk("The UE shall support NR-U operations in the unlicensed band.", "5.2.1"),
        _chunk("HARQ processes shall be scheduled per ServCellGroupId.", "5.2.2"),
    ]
    embedder = FakeEmbedder()
    store.build(chunks, embedder, source_fingerprint="fp1")

    query_vec = embedder.encode(["Tell me about NR-U unlicensed band operation"])[0]
    hits = store.search(query_vec, top_n=2)
    assert hits[0].metadata.clause_number == "5.2.1"


def test_persisted_index_is_loaded_without_rebuilding(tmp_path, monkeypatch):
    """This is the core regression test for 'FAISS index rebuilt on every
    start' — the chunks_provider callable must NOT be invoked when a fresh
    index is already on disk."""
    index_path = tmp_path / "index" / "faiss.index"
    meta_path = tmp_path / "meta" / "chunks.json"
    embedder = FakeEmbedder()

    store1 = VectorStore(index_path, meta_path)
    store1.build([_chunk("Some clause text.", "5.1")], embedder, source_fingerprint="fp1")

    calls = {"count": 0}

    def provider_should_not_be_called():
        calls["count"] += 1
        return [_chunk("Should not be reached.", "9.9")]

    store2 = VectorStore(index_path, meta_path)
    store2.load_or_build(provider_should_not_be_called, embedder, source_fingerprint="fp1")

    assert calls["count"] == 0
    assert store2.index.ntotal == 1
    assert store2.chunks[0].metadata.clause_number == "5.1"


def test_stale_fingerprint_triggers_rebuild(tmp_path):
    index_path = tmp_path / "index" / "faiss.index"
    meta_path = tmp_path / "meta" / "chunks.json"
    embedder = FakeEmbedder()

    store1 = VectorStore(index_path, meta_path)
    store1.build([_chunk("Old content.", "5.1")], embedder, source_fingerprint="fp_old")

    calls = {"count": 0}

    def provider():
        calls["count"] += 1
        return [_chunk("New content after PDFs changed.", "6.1")]

    store2 = VectorStore(index_path, meta_path)
    store2.load_or_build(provider, embedder, source_fingerprint="fp_new")

    assert calls["count"] == 1
    assert store2.chunks[0].metadata.clause_number == "6.1"


def test_force_rebuild_always_calls_provider_even_when_fresh(tmp_path):
    index_path = tmp_path / "index" / "faiss.index"
    meta_path = tmp_path / "meta" / "chunks.json"
    embedder = FakeEmbedder()

    store1 = VectorStore(index_path, meta_path)
    store1.build([_chunk("Original.", "5.1")], embedder, source_fingerprint="fp1")

    calls = {"count": 0}

    def provider():
        calls["count"] += 1
        return [_chunk("Rebuilt.", "5.1")]

    store2 = VectorStore(index_path, meta_path)
    store2.load_or_build(provider, embedder, source_fingerprint="fp1", force_rebuild=True)

    assert calls["count"] == 1


def test_fingerprint_changes_when_a_pdf_is_added(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-fake-a")
    fp1 = fingerprint_pdf_dir(tmp_path)
    (tmp_path / "b.pdf").write_bytes(b"%PDF-fake-b")
    fp2 = fingerprint_pdf_dir(tmp_path)
    assert fp1 != fp2
