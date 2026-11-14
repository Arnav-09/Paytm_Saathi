"""Custom exceptions for the Paytm Saathi AI engine.

Every engine error derives from PaytmSaathiError and carries an error_code so
the FastAPI layer (owned by another team member) can map exceptions to HTTP
responses in one place.
"""

from __future__ import annotations


class PaytmSaathiError(Exception):
    """Base exception for all AI engine errors."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "paytm_saathi_error"


class InvalidFileError(PaytmSaathiError):
    """Raised when an uploaded file is not a valid supported file type."""

    def __init__(self, message: str):
        super().__init__(message, "invalid_file")


class EmptyPDFError(PaytmSaathiError):
    """Raised when a PDF has no pages or no readable content."""

    def __init__(self, message: str = "The uploaded PDF is empty."):
        super().__init__(message, "empty_pdf")


class ScannedPDFError(PaytmSaathiError):
    """Raised when a PDF is image-based and requires OCR before processing."""

    def __init__(
        self,
        message: str = (
            "The uploaded PDF appears to be scanned or image-based. "
            "OCR is required before text extraction."
        ),
    ):
        super().__init__(message, "scanned_pdf")


class PDFExtractionError(PaytmSaathiError):
    """Raised when PDF text extraction fails."""

    def __init__(self, message: str):
        super().__init__(message, "pdf_extraction_error")


class LLMProviderError(PaytmSaathiError):
    """Raised when the LLM provider fails or is misconfigured."""

    def __init__(self, message: str):
        super().__init__(message, "llm_provider_error")


class EmbeddingError(PaytmSaathiError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str):
        super().__init__(message, "embedding_error")


class VectorStoreError(PaytmSaathiError):
    """Raised when ChromaDB operations fail."""

    def __init__(self, message: str):
        super().__init__(message, "vector_store_error")


class PolicyNotFoundError(PaytmSaathiError):
    """Raised when a policy_id is not found in the vector store or details store."""

    def __init__(self, policy_id: str):
        super().__init__(f"Policy not found: {policy_id}", "policy_not_found")


class ValidationError(PaytmSaathiError):
    """Raised when Pydantic validation of LLM-generated output fails."""

    def __init__(self, message: str):
        super().__init__(message, "validation_error")


class InvalidQuestionError(PaytmSaathiError):
    """Raised when the user question is empty or invalid."""

    def __init__(self, message: str):
        super().__init__(message, "invalid_question")


class OCRUnavailableError(PaytmSaathiError):
    """Raised when OCR is needed but no OCR backend is installed."""

    def __init__(
        self,
        message: str = (
            "OCR is required but no OCR engine is installed. "
            "Run: pip install rapidocr-onnxruntime   (or install Tesseract + pytesseract)."
        ),
    ):
        super().__init__(message, "ocr_unavailable")


class BillExtractionError(PaytmSaathiError):
    """Raised when a hospital bill cannot be read or contains no line items."""

    def __init__(self, message: str):
        super().__init__(message, "bill_extraction_error")


class CalculationInputError(PaytmSaathiError):
    """Raised when inputs to a financial calculation are invalid."""

    def __init__(self, message: str):
        super().__init__(message, "calculation_input_error")


class EvidenceRejectedError(PaytmSaathiError):
    """Evidence (SMS / screenshot / PDF) did not pass verification against the ticket and the user."""

    def __init__(self, reasons: list[str], reasons_hi: list[str] | None = None):
        self.reasons = reasons
        self.reasons_hi = reasons_hi or reasons
        self.message_hi = "सबूत स्वीकार नहीं हुआ: " + " ".join(self.reasons_hi)
        super().__init__("Evidence rejected: " + " ".join(reasons), "evidence_rejected")
