"""Tests for document_processor.py (Feature 1).

These tests create small PDFs in-memory with PyMuPDF, so they do not depend on
any sample files shipped with the repo.
"""

from __future__ import annotations

import pymupdf as fitz  # PyMuPDF
import pytest

from ai_engine.document_processor import extract_pdf_pages, validate_pdf_file
from ai_engine.exceptions import (
    EmptyPDFError,
    InvalidFileError,
    PDFExtractionError,
    ScannedPDFError,
)


def _make_pdf(tmp_path, pages_text, name="sample.pdf"):
    path = tmp_path / name
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def test_extract_preserves_page_numbers_and_filename(tmp_path):
    path = _make_pdf(tmp_path, ["Page one content.", "Page two content."])
    result = extract_pdf_pages(path)

    assert result.total_pages == 2
    assert result.has_extractable_text is True
    assert result.is_scanned is False
    assert result.source_filename == "sample.pdf"

    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "Page one content" in result.pages[0].page_text
    assert result.pages[0].source_filename == "sample.pdf"

    assert result.pages[1].page_number == 2
    assert "Page two content" in result.pages[1].page_text


def test_clean_whitespace_without_destroying_wording():
    from ai_engine.document_processor import clean_policy_text

    assert clean_policy_text("  Section   4.2   Room   Rent  ") == "Section 4.2 Room Rent"
    assert clean_policy_text("Line one\n\n\n\nLine two") == "Line one\n\nLine two"
    assert clean_policy_text("") == ""


def test_invalid_file_type_rejected(tmp_path):
    text_file = tmp_path / "note.txt"
    text_file.write_text("not a pdf")
    with pytest.raises(InvalidFileError):
        validate_pdf_file(text_file)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(InvalidFileError):
        validate_pdf_file(tmp_path / "does_not_exist.pdf")


def test_zero_byte_pdf_rejected(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(EmptyPDFError):
        extract_pdf_pages(empty)


def test_pdf_with_no_pages_raises(tmp_path):
    # PyMuPDF refuses to *save* a zero-page document, so write a minimal one by hand.
    path = tmp_path / "no_pages.pdf"
    path.write_bytes(
        b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj\n"
        b"trailer << /Root 1 0 R >>\n%%EOF\n"
    )
    with pytest.raises(EmptyPDFError):
        extract_pdf_pages(path)


def test_scanned_pdf_raises_instead_of_silent_empty(tmp_path):
    """PDF with pages but no extractable text -> ScannedPDFError."""
    path = _make_pdf(tmp_path, ["", ""])
    with pytest.raises(ScannedPDFError):
        extract_pdf_pages(path)


def test_corrupt_pdf_raises_pdf_extraction_error(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a pdf at all")
    with pytest.raises(PDFExtractionError):
        extract_pdf_pages(path)
