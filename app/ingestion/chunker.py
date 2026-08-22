"""
Turns a `ParsedDocument` (from pdf_parser.py) into retrieval-ready chunks.

Each chunk carries full citation metadata: spec_id, version, release,
clause number + title, and the page range it came from — so an answer
can be cited as e.g. "TS 38.331 v17.6.0, Clause 5.3.5.3, p.42" instead of
the old bare `[spec_id, Clause X]` string.

Long clauses are further split into overlapping windows so no single
chunk is too large for the embedding model's effective context, while
every sub-chunk still points back to its parent clause for citation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.ingestion.pdf_parser import ParsedDocument, PageLine, SpecMetadata, _HEADING_RE

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 20


@dataclass
class ChunkMetadata:
    spec_id: str
    doc_type: str
    spec_number: str
    version: str
    release: str
    clause_number: str
    clause_title: str
    page_start: int
    page_end: int
    source_file: str
    part_index: int = 0  # which sub-split of a long clause this is (0-indexed)
    part_count: int = 1

    def citation(self) -> str:
        loc = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        version_str = f" v{self.version}" if self.version != "unknown" else ""
        return f"{self.spec_id}{version_str}, Clause {self.clause_number} ({self.clause_title}), {loc}"

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "doc_type": self.doc_type,
            "spec_number": self.spec_number,
            "version": self.version,
            "release": self.release,
            "clause_number": self.clause_number,
            "clause_title": self.clause_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_file": self.source_file,
            "part_index": self.part_index,
            "part_count": self.part_count,
            "citation": self.citation(),
        }

    @staticmethod
    def from_dict(d: dict) -> "ChunkMetadata":
        d = dict(d)
        d.pop("citation", None)
        return ChunkMetadata(**d)


@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata


@dataclass
class _RawClause:
    number: str
    title: str
    body_lines: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


def _group_into_clauses(lines: list[PageLine]) -> list[_RawClause]:
    """Walk the page lines and group them under the heading that precedes them."""
    clauses: list[_RawClause] = []
    current: _RawClause | None = None

    for line in lines:
        heading_match = _HEADING_RE.match(line.text) if line.is_heading_candidate else None

        if heading_match:
            if current is not None:
                clauses.append(current)
            current = _RawClause(
                number=heading_match.group("number"),
                title=heading_match.group("title").strip(),
                page_start=line.page_number,
                page_end=line.page_number,
            )
        else:
            if current is None:
                # Body text before the first detected heading — group under an
                # implicit "0 Front matter" clause rather than dropping it.
                current = _RawClause(number="0", title="Front matter", page_start=line.page_number, page_end=line.page_number)
            current.body_lines.append(line.text)
            current.page_end = line.page_number

    if current is not None:
        clauses.append(current)

    return [c for c in clauses if c.body_lines]  # drop headings with no body at all


def _split_long_body(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        # try to break on a sentence boundary near the end
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period > start + MIN_CHUNK_CHARS:
                end = last_period + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return [p for p in parts if len(p) >= MIN_CHUNK_CHARS]


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Full pipeline: group lines under headings, then sub-split long clauses."""
    raw_clauses = _group_into_clauses(doc.lines)
    meta: SpecMetadata = doc.metadata

    chunks: list[Chunk] = []
    for clause in raw_clauses:
        body = " ".join(l for l in clause.body_lines if l).strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue

        parts = _split_long_body(body)
        for i, part_text in enumerate(parts):
            chunk_meta = ChunkMetadata(
                spec_id=meta.spec_id,
                doc_type=meta.doc_type,
                spec_number=meta.spec_number,
                version=meta.version,
                release=meta.release,
                clause_number=clause.number,
                clause_title=clause.title,
                page_start=clause.page_start or 0,
                page_end=clause.page_end or 0,
                source_file=meta.source_file,
                part_index=i,
                part_count=len(parts),
            )
            chunks.append(Chunk(content=part_text, metadata=chunk_meta))

    logger.info(
        "Chunked %s into %d chunks across %d clauses",
        meta.source_file,
        len(chunks),
        len(raw_clauses),
    )
    return chunks


def attach_tables_as_chunks(doc: ParsedDocument) -> list[Chunk]:
    """Represent extracted tables as their own citable chunks (rendered as
    Markdown) so numeric/tabular facts — e.g. MCS index tables, timer value
    tables — are retrievable and citable just like prose clauses."""
    meta = doc.metadata
    table_chunks: list[Chunk] = []

    for table in doc.tables:
        rows = table["rows"]
        if not rows:
            continue
        header, *body_rows = rows
        header = [str(h or "").strip() for h in header]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in body_rows:
            cells = [str(c or "").strip() for c in row]
            md_lines.append("| " + " | ".join(cells) + " |")
        content = "\n".join(md_lines)
        if len(content) < MIN_CHUNK_CHARS:
            continue

        chunk_meta = ChunkMetadata(
            spec_id=meta.spec_id,
            doc_type=meta.doc_type,
            spec_number=meta.spec_number,
            version=meta.version,
            release=meta.release,
            clause_number=f"Table@p{table['page']}",
            clause_title="Table",
            page_start=table["page"],
            page_end=table["page"],
            source_file=meta.source_file,
        )
        table_chunks.append(Chunk(content=content, metadata=chunk_meta))

    return table_chunks
