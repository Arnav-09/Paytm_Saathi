"""Feature 3: Document chunking.

Splits policy text into meaningful chunks, preferring section and clause
boundaries. Every chunk carries metadata (page number, section, clause
reference, source file) so retrieval results stay traceable to the PDF.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .schemas import Chunk, PageContent
from .utils import CHUNK_OVERLAP, CHUNK_SIZE, stable_chunk_id, logger

# Matches clause-like headings: "4.2 Room Rent Limit"
_CLAUSE_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Za-z][^\n]{2,70})$")
# Matches "Section 4", "Clause 5.1", "Chapter II", etc.
_SECTION_HEADING_RE = re.compile(
    r"^(section|clause|chapter|part|article)\s+([0-9IVXLCDM]+[.\w]*)\b[:\-–]?\s*(.+)?$",
    re.IGNORECASE,
)


def _detect_heading(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (section_name, clause_reference) from the first heading-like line.

    Only the first 15 non-empty lines are inspected, so body text is never
    mistaken for a heading.
    """
    for line in [ln.strip() for ln in text.splitlines()[:15] if ln.strip()]:
        match = _CLAUSE_HEADING_RE.match(line)
        if match:
            return match.group(2).strip(), match.group(1)
        match = _SECTION_HEADING_RE.match(line)
        if match:
            name = (match.group(3) or match.group(2)).strip()
            return name, match.group(2)
    return None, None


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into chunks of ~chunk_size characters.

    Cuts prefer paragraph boundaries, then sentence boundaries, so important
    conditions are not torn from the clause that qualifies them.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind("\n\n", start, end)
            if cut == -1:
                cut = text.rfind(". ", start, end)
            if cut != -1 and cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = end - overlap
        # Guarantee progress even when overlap is large.
        if start <= 0:
            start = end
    return pieces


def chunk_pages(
    pages: List[PageContent],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_filename: Optional[str] = None,
) -> List[Chunk]:
    """Chunk extracted pages, attaching page/section/clause metadata to each chunk."""
    chunks: List[Chunk] = []
    for page in pages:
        if not page.page_text.strip():
            continue
        file_name = source_filename or page.source_filename
        section_name, clause_reference = _detect_heading(page.page_text)
        pieces = _split_text(page.page_text, chunk_size, overlap)
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(file_name, page.page_number, index),
                    text=piece,
                    page_number=page.page_number,
                    section_name=section_name,
                    clause_reference=clause_reference,
                    source_filename=file_name,
                )
            )
    logger.info(
        "Created %d chunks from %d pages (file=%s)", len(chunks), len(pages), file_name
    )
    return chunks
