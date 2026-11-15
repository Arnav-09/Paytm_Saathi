"""Feature 1: Policy PDF processing.

Extracts text page-by-page from a health insurance policy PDF using PyMuPDF,
preserving page numbers. Detects scanned/image-based PDFs and reports that OCR
is required instead of silently returning empty text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pymupdf as fitz  # PyMuPDF

from .exceptions import (
    EmptyPDFError,
    InvalidFileError,
    PDFExtractionError,
    ScannedPDFError,
)
from .ocr import ocr_available, ocr_pdf_pages
from .schemas import DocumentExtractionResult, PageContent
from .utils import clean_policy_text, logger

SUPPORTED_EXTENSIONS = {".pdf"}
# A page with less than this many characters is treated as having no text.
MIN_TEXT_PER_PAGE = 1


def validate_pdf_file(file_path: Union[str, Path]) -> Path:
    """Check that the uploaded file exists, is non-empty, and is a PDF.

    Security: only .pdf files are accepted; uploaded content is never executed.
    """
    path = Path(file_path)
    if not path.exists():
        raise InvalidFileError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidFileError(
            f"Only PDF files are supported. Received file type: {path.suffix or 'none'}"
        )
    if path.stat().st_size == 0:
        raise EmptyPDFError("The uploaded PDF file is empty (0 bytes).")
    return path


def extract_pdf_pages(
    file_path: Union[str, Path], allow_ocr: bool = True
) -> DocumentExtractionResult:
    """Extract text from every page of a policy PDF.

    Returns page-level content (page_number, page_text, source_filename).
    When the PDF has no text layer (a scan) and ``allow_ocr`` is true, pages are
    OCRed via ``ocr.py`` if an OCR backend is installed. Raises ScannedPDFError
    when no text can be obtained at all, so an image-only PDF is never silently
    treated as an empty document.
    """
    path = validate_pdf_file(file_path)

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # PyMuPDF raises various errors for corrupt files
        raise PDFExtractionError(f"Could not open PDF file: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise EmptyPDFError("The PDF contains no pages.")

        pages: list[PageContent] = []
        all_text = ""

        for index in range(doc.page_count):
            page = doc[index]
            try:
                raw_text = page.get_text("text") or ""
            except Exception as exc:
                logger.warning("Failed to extract text from page %d: %s", index + 1, exc)
                raw_text = ""

            page_text = clean_policy_text(raw_text)
            all_text += page_text + "\n"

            pages.append(
                PageContent(
                    page_number=index + 1,
                    page_text=page_text,
                    source_filename=path.name,
                )
            )

        has_extractable_text = len(all_text.strip()) > 0
        if not has_extractable_text:
            ocr_pages = _try_ocr(path) if allow_ocr else None
            if not ocr_pages:
                raise ScannedPDFError(
                    "No extractable text found in the PDF. It appears to be scanned or "
                    "image-based, and OCR "
                    + ("produced no text. Try a clearer scan."
                       if allow_ocr and ocr_available()
                       else "is not installed. Run: pip install rapidocr-onnxruntime")
                )
            return DocumentExtractionResult(
                source_filename=path.name,
                total_pages=doc.page_count,
                pages=ocr_pages,
                has_extractable_text=True,
                is_scanned=True,
            )

        return DocumentExtractionResult(
            source_filename=path.name,
            total_pages=doc.page_count,
            pages=pages,
            has_extractable_text=True,
            is_scanned=False,
        )
    finally:
        doc.close()


def _try_ocr(path: Path) -> list[PageContent]:
    """OCR a scanned PDF. Returns [] when OCR is unavailable or finds nothing."""
    if not ocr_available():
        return []
    try:
        raw_pages = ocr_pdf_pages(path)
    except Exception as exc:  # OCR is best-effort; caller raises ScannedPDFError
        logger.warning("OCR failed: %s", exc)
        return []
    pages = [
        PageContent(
            page_number=number,
            page_text=clean_policy_text(text),
            source_filename=path.name,
        )
        for number, text in raw_pages
    ]
    return pages if any(page.page_text for page in pages) else []
