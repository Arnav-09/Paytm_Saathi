"""OCR for scanned policy PDFs and photographed hospital bills.

Backends are optional and tried in order:

1. ``rapidocr-onnxruntime`` - pure pip install, no system binary (recommended).
2. ``pytesseract`` + Pillow  - needs the Tesseract binary; supports Hindi with
   the ``hin`` language pack (``OCR_LANGS=eng+hin``).

If neither is installed, :class:`OCRUnavailableError` is raised with install
instructions, so a scanned document is never silently treated as empty.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union

from .exceptions import OCRUnavailableError, PDFExtractionError
from .utils import logger

OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_LANGS = os.getenv("OCR_LANGS", "eng")

_RAPID = None  # lazily created RapidOCR instance


def _get_rapid():
    global _RAPID
    if _RAPID is None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        _RAPID = RapidOCR()
    return _RAPID


def available_backend() -> Optional[str]:
    """Name of the first usable OCR backend, or None."""
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return "rapidocr"
    except Exception:
        pass
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
        return "tesseract"
    except Exception:
        return None


def ocr_available() -> bool:
    return available_backend() is not None


def _rapid_to_text(result) -> str:
    """Rebuild reading-order lines from RapidOCR [(box, text, score), ...]."""
    if not result:
        return ""
    items = []
    for box, text, _score in result:
        ys = [point[1] for point in box]
        xs = [point[0] for point in box]
        items.append(
            {
                "y": sum(ys) / len(ys),
                "x": min(xs),
                "h": max(max(ys) - min(ys), 1.0),
                "text": str(text).strip(),
            }
        )
    items.sort(key=lambda item: (item["y"], item["x"]))
    lines: list[list[dict]] = []
    for item in items:
        if lines and abs(item["y"] - lines[-1][0]["y"]) < 0.6 * lines[-1][0]["h"]:
            lines[-1].append(item)
        else:
            lines.append([item])
    out = []
    for line in lines:
        line.sort(key=lambda item: item["x"])
        out.append("  ".join(item["text"] for item in line if item["text"]))
    return "\n".join(out)


def ocr_image_bytes(data: bytes) -> str:
    """OCR one image (PNG/JPEG bytes) and return text in reading order."""
    backend = available_backend()
    if backend is None:
        raise OCRUnavailableError()
    try:
        if backend == "rapidocr":
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore

            image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            result, _elapsed = _get_rapid()(image)
            return _rapid_to_text(result)
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        image = Image.open(io.BytesIO(data)).convert("RGB")
        return pytesseract.image_to_string(image, lang=OCR_LANGS)
    except OCRUnavailableError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"OCR failed: {exc}") from exc


def ocr_pdf_pages(path: Union[str, Path]) -> list[tuple[int, str]]:
    """Render each PDF page to an image and OCR it. Returns [(page_number, text)]."""
    if not ocr_available():
        raise OCRUnavailableError()
    import pymupdf as fitz  # PyMuPDF

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise PDFExtractionError(f"Could not open PDF for OCR: {exc}") from exc
    pages: list[tuple[int, str]] = []
    try:
        for index in range(doc.page_count):
            png = doc[index].get_pixmap(dpi=OCR_DPI).tobytes("png")
            text = ocr_image_bytes(png)
            logger.info("OCR page %d/%d: %d chars", index + 1, doc.page_count, len(text))
            pages.append((index + 1, text))
    finally:
        doc.close()
    return pages
