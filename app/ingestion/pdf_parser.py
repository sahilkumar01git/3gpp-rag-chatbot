"""
PDF ingestion for real 3GPP specification documents.

This replaces the old approach of reading a single hand-written
`mock_3gpp.txt`. It works on actual `.pdf` files dropped into
`data/raw_pdfs/` (3GPP TS/TR documents downloaded from
https://www.3gpp.org/specifications-technologies, or any similarly
structured technical spec).

Two things make 3GPP PDFs awkward to parse naively:

1. Every page repeats a header/footer like
   "3GPP TS 38.331 version 17.6.0 Release 17" — useful for metadata,
   but noise if left inline with body text.
2. Section headings ("4.2.1  RRC connection establishment") are not
   marked up semantically in the PDF — they are just text that happens
   to be bold and/or larger than body text. We use pdfplumber's
   character-level font metadata to detect them, with a pure regex
   fallback for PDFs where font info isn't reliable (e.g. scanned/OCR'd
   text layers).
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# "3GPP TS 38.331 V17.6.0 (2023-12)"  or  "3GPP TS 38.331 version 17.6.0 Release 17"
_SPEC_HEADER_RE = re.compile(
    r"3GPP\s+(?P<doc_type>TS|TR)\s+(?P<spec_number>\d{1,2}\.\d{3})"
    r"[^\dVv]{0,15}[Vv](?:ersion)?\s*(?P<version>\d+\.\d+\.\d+)"
    r"(?:.*?Release\s*(?P<release>\d+))?",
    re.IGNORECASE,
)

# Numbered heading, e.g. "4.2.1 RRC connection establishment" (1-4 levels deep).
_HEADING_RE = re.compile(r"^(?P<number>\d{1,2}(?:\.\d{1,3}){0,4})\s+(?P<title>[A-Z][^\n]{2,120})$")

# Lines that are pure header/footer boilerplate we want to strip from body text.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^3GPP\s+(TS|TR)\s+\d{1,2}\.\d{3}.*$", re.IGNORECASE),
    re.compile(r"^Page\s+\d+\s*(of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^\d+$"),  # bare page numbers
]


@dataclass
class SpecMetadata:
    spec_id: str  # e.g. "TS 38.331"
    doc_type: str  # "TS" or "TR"
    spec_number: str  # "38.331"
    version: str = "unknown"
    release: str = "unknown"
    title: str = ""
    source_file: str = ""


@dataclass
class PageLine:
    text: str
    page_number: int  # 1-indexed
    is_heading_candidate: bool = False


@dataclass
class ParsedDocument:
    metadata: SpecMetadata
    lines: list[PageLine] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)  # [{page, index, rows}]


def _is_boilerplate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in _BOILERPLATE_PATTERNS)


def _extract_spec_metadata(first_pages_text: str, source_file: str) -> SpecMetadata:
    match = _SPEC_HEADER_RE.search(first_pages_text)
    if match:
        doc_type = match.group("doc_type").upper()
        number = match.group("spec_number")
        return SpecMetadata(
            spec_id=f"{doc_type} {number}",
            doc_type=doc_type,
            spec_number=number,
            version=match.group("version") or "unknown",
            release=match.group("release") or "unknown",
            source_file=source_file,
        )

    # Fallback: try to infer from the filename, e.g. "TS_38.331-h60.pdf" or "38331-h60.pdf"
    stem = Path(source_file).stem
    fname_match = re.search(r"(TS|TR)?[_\-\s]?(\d{2})[.\-]?(\d{3})", stem, re.IGNORECASE)
    if fname_match:
        doc_type = (fname_match.group(1) or "TS").upper()
        number = f"{fname_match.group(2)}.{fname_match.group(3)}"
        return SpecMetadata(
            spec_id=f"{doc_type} {number}",
            doc_type=doc_type,
            spec_number=number,
            source_file=source_file,
        )

    logger.warning("Could not detect 3GPP spec header in %s — using filename as spec_id", source_file)
    return SpecMetadata(
        spec_id=stem,
        doc_type="UNKNOWN",
        spec_number="unknown",
        source_file=source_file,
    )


def _font_size_heading_detector(page) -> set[str]:
    """Returns the set of line-texts on this page whose average character
    font size is meaningfully larger than the page's median (i.e. likely
    headings). Falls back to an empty set if font metadata is unavailable."""
    try:
        chars = page.chars
    except Exception:  # pragma: no cover - pdfplumber internal edge cases
        return set()

    if not chars:
        return set()

    sizes = [c["size"] for c in chars if c.get("size")]
    if not sizes:
        return set()

    median_size = statistics.median(sizes)

    # Group characters into lines by their `top` (vertical) position.
    lines: dict[float, list[dict]] = {}
    for c in chars:
        key = round(c["top"], 0)
        lines.setdefault(key, []).append(c)

    heading_texts = set()
    for line_chars in lines.values():
        line_chars.sort(key=lambda c: c["x0"])
        text = "".join(c["text"] for c in line_chars).strip()
        if not text:
            continue
        avg_size = sum(c["size"] for c in line_chars if c.get("size")) / max(len(line_chars), 1)
        is_bold = any("bold" in (c.get("fontname") or "").lower() for c in line_chars)
        if avg_size > median_size * 1.08 or (is_bold and avg_size >= median_size):
            heading_texts.add(text)

    return heading_texts


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse a single 3GPP PDF into per-page lines (with heading flags) and tables."""
    logger.info("Parsing PDF: %s", path)

    all_lines: list[PageLine] = []
    all_tables: list[dict] = []
    first_pages_text_parts: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_index <= 3:
                first_pages_text_parts.append(page_text)

            heading_candidates = _font_size_heading_detector(page)

            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if _is_boilerplate(line):
                    continue
                is_heading = line in heading_candidates or bool(_HEADING_RE.match(line))
                all_lines.append(PageLine(text=line, page_number=page_index, is_heading_candidate=is_heading))

            try:
                for t_index, table in enumerate(page.extract_tables()):
                    if table and any(any(cell for cell in row) for row in table):
                        all_tables.append({"page": page_index, "index": t_index, "rows": table})
            except Exception as exc:  # pragma: no cover
                logger.debug("Table extraction failed on page %s of %s: %s", page_index, path, exc)

    metadata = _extract_spec_metadata("\n".join(first_pages_text_parts), source_file=path.name)

    # Best-effort title: first reasonably long line on page 1 that isn't the spec header itself.
    for line in all_lines[:40]:
        if len(line.text) > 15 and "3GPP" not in line.text and not _HEADING_RE.match(line.text):
            metadata.title = line.text
            break

    logger.info(
        "Parsed %s: spec_id=%s version=%s pages=%s lines=%s tables=%s",
        path.name,
        metadata.spec_id,
        metadata.version,
        (all_lines[-1].page_number if all_lines else 0),
        len(all_lines),
        len(all_tables),
    )

    return ParsedDocument(metadata=metadata, lines=all_lines, tables=all_tables)


def discover_pdfs(raw_pdf_dir: Path) -> list[Path]:
    """Recursively find all PDFs under the raw PDF directory (including the
    bundled synthetic sample corpus subfolder)."""
    if not raw_pdf_dir.exists():
        return []
    return sorted(raw_pdf_dir.rglob("*.pdf"))
