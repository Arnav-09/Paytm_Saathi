"""Paytm Saathi - AI and RAG engine for health insurance claim pre-flight.

This package is an independent, modular Python package. The backend team
(FastAPI) can integrate through these function calls, and the frontend team
(Streamlit) can consume their return values directly.

Public API:
    process_policy(pdf_path, policy_id)
    ask_policy_question(policy_id, question, conversation_id=None)
    get_policy_details(policy_id)
    explain_policy_clause(policy_id, clause_reference)
    get_claim_relevant_clauses(policy_id, treatment_details=None)
"""

from __future__ import annotations

from typing import Optional

from .document_processor import extract_pdf_pages
from .embeddings import get_embedding_model
from .exceptions import (
    EmptyPDFError,
    EmbeddingError,
    InvalidFileError,
    InvalidQuestionError,
    LLMProviderError,
    PaytmSaathiError,
    PDFExtractionError,
    PolicyNotFoundError,
    ScannedPDFError,
    ValidationError,
    VectorStoreError,
)
from .llm_provider import (
    LLMProvider,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    get_llm_provider,
)
from .policy_extractor import (
    PolicyExtractor,
    load_policy_details,
    save_policy_details,
)
from .prompts import (
    MANDATORY_RAG_INSTRUCTIONS,
    claim_clauses_prompt,
    follow_up_prompt,
    policy_qa_prompt,
    simplification_prompt,
    structured_extraction_prompt,
)
from .rag_engine import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    RAGEngine,
)
from .retriever import PolicyRetriever, RetrievedChunk
from .schemas import (
    ClaimRelevantClause,
    ClaimRelevantClausesResult,
    Chunk,
    ConfidenceLevel,
    CoverageCondition,
    DocumentExtractionResult,
    ExclusionEntry,
    MoneyValue,
    PageContent,
    PolicyDetails,
    RAGResponse,
    SimpleExplanation,
    SourceCitation,
    SubLimitEntry,
    WaitingPeriodEntry,
)
from .text_chunker import chunk_pages
from .vector_store import ChromaVectorStore

__all__ = [
    "process_policy",
    "ask_policy_question",
    "get_policy_details",
    "explain_policy_clause",
    "get_claim_relevant_clauses",
    "extract_pdf_pages",
    "chunk_pages",
    "get_embedding_model",
    "get_llm_provider",
    "RAGEngine",
    "PolicyRetriever",
    "PolicyExtractor",
    "ChromaVectorStore",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "RAGResponse",
    "PolicyDetails",
    "SimpleExplanation",
    "ClaimRelevantClausesResult",
    "SourceCitation",
    "PageContent",
    "Chunk",
    "DocumentExtractionResult",
    "MoneyValue",
    "WaitingPeriodEntry",
    "ExclusionEntry",
    "SubLimitEntry",
    "CoverageCondition",
    "ConfidenceLevel",
    "RetrievedChunk",
    "PaytmSaathiError",
    "InvalidFileError",
    "EmptyPDFError",
    "ScannedPDFError",
    "PDFExtractionError",
    "LLMProviderError",
    "EmbeddingError",
    "VectorStoreError",
    "PolicyNotFoundError",
    "ValidationError",
    "InvalidQuestionError",
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "MANDATORY_RAG_INSTRUCTIONS",
]

# Module-level singletons reused across calls.
_ENGINE: Optional[RAGEngine] = None


def _get_engine() -> RAGEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RAGEngine()
    return _ENGINE


def process_policy(pdf_path: str, policy_id: str) -> PolicyDetails:
    """End-to-end policy ingestion pipeline.

    1. Extracts PDF text page by page (PyMuPDF).
    2. Chunks the document preserving page/section/clause metadata.
    3. Generates embeddings (Sentence Transformers).
    4. Stores chunks in ChromaDB (skipped if already ingested).
    5. Extracts structured policy details (LLM + Pydantic validation).

    Returns the structured PolicyDetails object.
    """
    result = extract_pdf_pages(pdf_path)
    pages = result.pages
    if not pages:
        raise EmptyPDFError("No pages were extracted from the PDF.")

    chunks = chunk_pages(pages, source_filename=result.source_filename)
    if not chunks:
        raise EmptyPDFError("No text chunks could be created from the PDF.")

    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_texts([chunk.text for chunk in chunks])

    store = ChromaVectorStore()
    store.add_chunks(policy_id, chunks, embeddings)

    extractor = PolicyExtractor()
    details = extractor.extract(pages, source_filename=result.source_filename)
    save_policy_details(policy_id, details)

    return details


def ask_policy_question(
    policy_id: str, question: str, conversation_id: Optional[str] = None
) -> RAGResponse:
    """Answer a question about a policy, with citations and conversation context."""
    return _get_engine().ask(policy_id, question, conversation_id)


def get_policy_details(policy_id: str) -> PolicyDetails:
    """Return previously extracted structured policy information."""
    return load_policy_details(policy_id)


def explain_policy_clause(policy_id: str, clause_reference: str) -> SimpleExplanation:
    """Retrieve a clause and explain it in simple language."""
    return _get_engine().explain_policy_clause(policy_id, clause_reference)


def get_claim_relevant_clauses(
    policy_id: str, treatment_details: Optional[dict] = None
) -> ClaimRelevantClausesResult:
    """Return claim-relevant clauses and structured conditions for the calculation backend."""
    return _get_engine().get_claim_relevant_clauses(policy_id, treatment_details)
