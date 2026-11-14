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

from .bill_extractor import extract_bill, extract_bill_from_text, parse_bill_text
from .document_processor import extract_pdf_pages
from .language import (
    detect_language,
    normalize_language,
    question_to_english,
    translate_text,
)
from .payout_calculator import estimate_payout
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
    list_policy_ids,
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
    ClaimContext,
    HospitalBill,
    PayoutEstimate,
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
    "explain_clause_text",
    "get_claim_relevant_clauses",
    "extract_hospital_bill",
    "estimate_claim_payout",
    "list_policy_ids",
    "translate_text",
    "detect_language",
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
    policy_id: str,
    question: str,
    conversation_id: Optional[str] = None,
    language: Optional[str] = None,
) -> RAGResponse:
    """Answer a question about a policy, with citations and conversation context.

    ``language`` is a code: "en", "hi", "mr", "bn", "ta", "te" or "gu". When omitted it is
    detected from the question's script, so a Hindi question gets a Hindi answer. Retrieval
    always runs in English; citations keep
    the policy's original wording. If translation cannot be done safely (numbers or
    citations would change) the English answer is returned with ``language="en"``.
    """
    engine = _get_engine()
    target = normalize_language(language) if language else detect_language(question)
    english_question = question_to_english(question, engine.llm) if target != "en" else question
    response = engine.ask(policy_id, english_question, conversation_id)
    response.question = question  # echo what the user actually typed
    if target != "en":
        translated, ok = translate_text(response.answer, target, engine.llm)
        if ok:
            response.answer_en, response.answer, response.language = response.answer, translated, target
    return response


def get_policy_details(policy_id: str) -> PolicyDetails:
    """Return previously extracted structured policy information."""
    return load_policy_details(policy_id)


def explain_policy_clause(
    policy_id: str, clause_reference: str, language: Optional[str] = "en"
) -> SimpleExplanation:
    """Retrieve a clause and explain it in simple language (English or Hindi)."""
    engine = _get_engine()
    result = engine.explain_policy_clause(policy_id, clause_reference)
    target = normalize_language(language)
    if target != "en":
        translated, ok = translate_text(result.simple_explanation, target, engine.llm)
        if ok:
            result.simple_explanation, result.language = translated, target
    return result


def get_claim_relevant_clauses(
    policy_id: str, treatment_details: Optional[dict] = None
) -> ClaimRelevantClausesResult:
    """Return claim-relevant clauses and structured conditions for the calculation backend."""
    return _get_engine().get_claim_relevant_clauses(policy_id, treatment_details)


def extract_hospital_bill(file_path: str) -> HospitalBill:
    """Read a hospital bill (PDF, scanned PDF or photo) into structured line items."""
    return extract_bill(file_path, llm=_get_engine().llm)


def estimate_claim_payout(
    policy_id: str, bill: HospitalBill, context: Optional[ClaimContext] = None
) -> PayoutEstimate:
    """Deterministic payout estimate for a bill under an ingested policy.

    Uses the policy details extracted earlier and pure Python arithmetic (no LLM).
    """
    details = load_policy_details(policy_id)
    return estimate_payout(details, bill, context, policy_id=policy_id)


def explain_clause_text(text: str, language: Optional[str] = "en") -> SimpleExplanation:
    """Explain pasted policy wording in simple language (no clause number needed)."""
    if not text or len(text.strip()) < 20:
        raise InvalidQuestionError("Paste at least one full sentence of the policy wording.")
    engine = _get_engine()
    original = text.strip()[:4000]
    if isinstance(engine.llm, MockLLMProvider):
        explanation = INSUFFICIENT_EVIDENCE_ANSWER
    else:
        explanation = engine.llm.generate(
            simplification_prompt(original), system_prompt=MANDATORY_RAG_INSTRUCTIONS).strip()
    result = SimpleExplanation(policy_id="pasted_text", original_clause=original,
                               simple_explanation=explanation)
    target = normalize_language(language)
    if target != "en":
        translated, ok = translate_text(result.simple_explanation, target, engine.llm)
        if ok:
            result.simple_explanation, result.language = translated, target
    return result
