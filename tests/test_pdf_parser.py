from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.ingestion.pdf_parser import discover_pdfs, parse_pdf


def _make_test_pdf(path: Path) -> None:
    """Builds a tiny but real multi-page PDF with a 3GPP-style running
    header/footer and numbered clause headings, so the parser is exercised
    against actual PDF bytes rather than pre-chunked strings."""
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    def header_footer(page_num: int):
        c.setFont("Helvetica", 8)
        c.drawString(72, height - 40, "3GPP TS 38.331 V17.6.0 (2023-12)")
        c.drawString(72, 30, f"Page {page_num}")

    # Page 1
    header_footer(1)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 100, "Radio Resource Control (RRC) protocol specification")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 140, "5.2.1 General")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 160, "The UE shall support NR-U operations in the specified band.")
    c.showPage()

    # Page 2 — a new clause plus a table
    header_footer(2)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 100, "5.2.2 PDSCH Scheduling")
    c.setFont("Helvetica", 10)
    c.drawString(72, height - 120, "The gNB shall schedule PDSCH transmissions via DCI format 0_1.")
    # A minimal table pdfplumber can detect (drawn as a simple grid of lines + text).
    table_top = height - 160
    rows = [["MCS", "Modulation"], ["0", "QPSK"], ["10", "16QAM"]]
    col_w, row_h = 100, 20
    for r, row in enumerate(rows):
        for col, cell in enumerate(row):
            c.rect(72 + col * col_w, table_top - r * row_h - row_h, col_w, row_h)
            c.drawString(72 + col * col_w + 5, table_top - r * row_h - row_h + 5, cell)
    c.showPage()
    c.save()


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    pdf_path = tmp_path / "ts_38331_sample.pdf"
    _make_test_pdf(pdf_path)
    return pdf_path


def test_extracts_spec_metadata_from_running_header(sample_pdf):
    doc = parse_pdf(sample_pdf)
    assert doc.metadata.spec_id == "TS 38.331"
    assert doc.metadata.version == "17.6.0"


def test_strips_header_footer_boilerplate_from_body_lines(sample_pdf):
    doc = parse_pdf(sample_pdf)
    joined = " ".join(l.text for l in doc.lines)
    assert "3GPP TS 38.331" not in joined
    assert "Page 1" not in joined
    assert "Page 2" not in joined


def test_tracks_page_numbers_across_pages(sample_pdf):
    doc = parse_pdf(sample_pdf)
    pages = {l.page_number for l in doc.lines}
    assert pages == {1, 2}


def test_detects_heading_candidates(sample_pdf):
    doc = parse_pdf(sample_pdf)
    heading_lines = [l.text for l in doc.lines if l.is_heading_candidate]
    assert any("5.2.1" in t for t in heading_lines)
    assert any("5.2.2" in t for t in heading_lines)


def test_extracts_table(sample_pdf):
    doc = parse_pdf(sample_pdf)
    assert len(doc.tables) >= 1
    flat_cells = [cell for table in doc.tables for row in table["rows"] for cell in row if cell]
    assert any("QPSK" in c for c in flat_cells)


def test_discover_pdfs_finds_files_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    _make_test_pdf(tmp_path / "a.pdf")
    _make_test_pdf(tmp_path / "sub" / "b.pdf")
    found = discover_pdfs(tmp_path)
    assert len(found) == 2


def test_discover_pdfs_empty_dir_returns_empty_list(tmp_path):
    assert discover_pdfs(tmp_path / "does_not_exist") == []
