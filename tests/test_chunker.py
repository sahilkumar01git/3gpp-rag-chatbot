from app.ingestion.chunker import MAX_CHUNK_CHARS, chunk_document
from app.ingestion.pdf_parser import ParsedDocument, PageLine, SpecMetadata


def _doc(lines: list[tuple[str, int, bool]]) -> ParsedDocument:
    meta = SpecMetadata(spec_id="TS 38.331", doc_type="TS", spec_number="38.331", version="17.6.0", release="17", source_file="ts_38331.pdf")
    page_lines = [PageLine(text=t, page_number=p, is_heading_candidate=h) for t, p, h in lines]
    return ParsedDocument(metadata=meta, lines=page_lines, tables=[])


def test_groups_body_text_under_preceding_heading():
    doc = _doc(
        [
            ("5.2.1 General", 1, True),
            ("The UE shall support NR-U operations.", 1, False),
            ("The maximum EIRP shall not exceed 20 dBm.", 1, False),
            ("5.2.2 PDSCH Scheduling", 2, True),
            ("The gNB shall schedule PDSCH via DCI format 0_1.", 2, False),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].metadata.clause_number == "5.2.1"
    assert "NR-U" in chunks[0].content
    assert chunks[1].metadata.clause_number == "5.2.2"
    assert "PDSCH" in chunks[1].content


def test_carries_full_citation_metadata():
    doc = _doc([("5.3.1 RLC Modes", 3, True), ("The UE shall support TM1 and TM2.", 3, False)])
    chunk = chunk_document(doc)[0]
    citation = chunk.metadata.citation()
    assert "TS 38.331" in citation
    assert "17.6.0" in citation
    assert "5.3.1" in citation
    assert "p.3" in citation


def test_body_text_before_any_heading_is_not_dropped():
    doc = _doc([("Some scope text with no heading yet that is long enough.", 1, False)])
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata.clause_number == "0"


def test_long_clause_is_split_with_overlap_and_shared_metadata():
    long_body = "This sentence repeats. " * 100  # comfortably over MAX_CHUNK_CHARS
    assert len(long_body) > MAX_CHUNK_CHARS
    doc = _doc([("6.1 Long Clause", 1, True), (long_body, 1, False)])
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(c.metadata.clause_number == "6.1" for c in chunks)
    assert chunks[0].metadata.part_count == len(chunks)


def test_page_range_spans_multi_page_clause():
    doc = _doc(
        [
            ("7.1 Cross Page Clause", 4, True),
            ("Text on page four.", 4, False),
            ("Text continues on page five.", 5, False),
        ]
    )
    chunk = chunk_document(doc)[0]
    assert chunk.metadata.page_start == 4
    assert chunk.metadata.page_end == 5
