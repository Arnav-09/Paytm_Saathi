"""Feature 6: RAG question answering.

The engine retrieves policy clauses, builds a context, asks the configured
LLM, and returns an answer with structured citations. Conversational context
is kept per conversation_id for follow-up questions.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .exceptions import (
    InvalidQuestionError,
    LLMProviderError,
    PolicyNotFoundError,
)
from .llm_provider import LLMProvider, get_llm_provider
from .prompts import (
    MANDATORY_RAG_INSTRUCTIONS,
    follow_up_prompt,
    policy_qa_prompt,
    simplification_prompt,
)
from .retriever import PolicyRetriever, RetrievedChunk
from .schemas import (
    ClaimRelevantClause,
    ClaimRelevantClausesResult,
    ConfidenceLevel,
    RAGResponse,
    SimpleExplanation,
    SourceCitation,
)
from .utils import logger
from .vector_store import ChromaVectorStore

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find enough information in the uploaded policy to answer this reliably."
)

CITATION_RE = re.compile(r"\[Source\s*(\d+)\]", re.IGNORECASE)


def _excerpt(text: str, max_chars: int = 260) -> str:
    """Return a short snippet of original policy wording for a citation."""
    text = text.strip().replace("\n", " ")
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    excerpt = parts[0] if parts else text
    if len(excerpt) < max_chars and len(parts) > 1:
        excerpt = (excerpt + " " + parts[1]).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0] + "..."
    return excerpt


def _confidence_level(chunks: List[RetrievedChunk], cited: List[RetrievedChunk]) -> ConfidenceLevel:
    if not chunks:
        return ConfidenceLevel.low
    avg_score = sum(chunk.score for chunk in chunks) / len(chunks)
    if cited and avg_score >= 0.60:
        return ConfidenceLevel.high
    if avg_score >= 0.40:
        return ConfidenceLevel.medium
    return ConfidenceLevel.low


class RAGEngine:
    def __init__(
        self,
        retriever: PolicyRetriever | None = None,
        llm_provider: LLMProvider | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.retriever = retriever or PolicyRetriever()
        self.llm = llm_provider or get_llm_provider()
        self.vector_store = vector_store or ChromaVectorStore()
        # In-memory conversation history (per conversation_id).
        self._conversations: dict[str, List[dict]] = {}

    # ------------------------------------------------------------------
    # Feature 6: question answering
    # ------------------------------------------------------------------
    def ask(
        self,
        policy_id: str,
        question: str,
        conversation_id: Optional[str] = None,
    ) -> RAGResponse:
        if not question or not question.strip():
            raise InvalidQuestionError("Question cannot be empty or whitespace.")

        chunks = self.retriever.retrieve(policy_id, question)

        if not chunks:
            response = RAGResponse(
                policy_id=policy_id,
                question=question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                sources=[],
                confidence=ConfidenceLevel.low,
                insufficient_evidence=True,
                conversation_id=conversation_id,
            )
            self._record_turn(conversation_id, question, response.answer)
            return response

        context_items = [
            {
                "index": index + 1,
                "page_number": chunk.page_number,
                "section_label": chunk.section_name or chunk.clause_reference or "policy clause",
                "clause_reference": chunk.clause_reference,
                "text": chunk.text,
            }
            for index, chunk in enumerate(chunks)
        ]

        history = self._conversations.get(conversation_id, []) if conversation_id else []
        prompt = (
            follow_up_prompt(question, context_items, history)
            if history
            else policy_qa_prompt(question, context_items)
        )

        try:
            answer_text = self.llm.generate(prompt, system_prompt=MANDATORY_RAG_INSTRUCTIONS)
        except LLMProviderError as exc:
            raise LLMProviderError(f"Failed to generate an answer: {exc}") from exc

        cited = self._parse_citations(answer_text, chunks)

        if INSUFFICIENT_EVIDENCE_ANSWER in answer_text or not cited:
            # Never fabricate citations when evidence is insufficient.
            response = RAGResponse(
                policy_id=policy_id,
                question=question,
                answer=answer_text or INSUFFICIENT_EVIDENCE_ANSWER,
                sources=[],
                confidence=ConfidenceLevel.low,
                insufficient_evidence=True,
                conversation_id=conversation_id,
            )
        else:
            sources = [
                SourceCitation(
                    page_number=chunk.page_number,
                    section_name=chunk.section_name,
                    clause_reference=chunk.clause_reference,
                    quoted_text=_excerpt(chunk.text),
                )
                for chunk in cited
            ]
            response = RAGResponse(
                policy_id=policy_id,
                question=question,
                answer=answer_text,
                sources=sources,
                confidence=_confidence_level(chunks, cited),
                insufficient_evidence=False,
                conversation_id=conversation_id,
            )

        self._record_turn(conversation_id, question, answer_text)
        return response

    @staticmethod
    def _parse_citations(
        answer_text: str, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Map [Source N] mentions in the answer back to retrieved chunks."""
        cited: List[RetrievedChunk] = []
        for match in CITATION_RE.finditer(answer_text):
            index = int(match.group(1)) - 1
            if 0 <= index < len(chunks) and chunks[index] not in cited:
                cited.append(chunks[index])
        return cited

    def _record_turn(
        self,
        conversation_id: Optional[str],
        question: str,
        answer: str,
    ) -> None:
        if not conversation_id:
            return
        history = self._conversations.setdefault(conversation_id, [])
        history.append({"question": question, "answer": answer})
        # Keep only the most recent turns so prompts stay bounded.
        if len(history) > 10:
            del history[: len(history) - 10]


    # ------------------------------------------------------------------
    # Feature 7: explain a clause in simple language
    # ------------------------------------------------------------------
    def explain_policy_clause(
        self, policy_id: str, clause_reference: str
    ) -> SimpleExplanation:
        if not clause_reference or not clause_reference.strip():
            raise InvalidQuestionError("clause_reference cannot be empty.")

        hits = self.vector_store.get_by_metadata(
            policy_id, {"clause_reference": clause_reference.strip()}, limit=1
        )
        if not hits:
            raise PolicyNotFoundError(
                f"Clause {clause_reference!r} not found for policy {policy_id}."
            )

        hit = hits[0]
        original_clause = hit["text"]

        try:
            simple_explanation = self.llm.generate(
                simplification_prompt(original_clause),
                system_prompt=MANDATORY_RAG_INSTRUCTIONS,
            )
        except LLMProviderError as exc:
            raise LLMProviderError(f"Failed to explain clause: {exc}") from exc

        return SimpleExplanation(
            policy_id=policy_id,
            clause_reference=clause_reference.strip(),
            page_number=hit["page_number"],
            original_clause=original_clause,
            simple_explanation=simple_explanation.strip(),
            source_filename=hit["source_filename"],
        )

    # ------------------------------------------------------------------
    # Feature 9: claim pre-flight support
    # ------------------------------------------------------------------
    def get_claim_relevant_clauses(
        self, policy_id: str, treatment_details: Optional[dict] = None
    ) -> ClaimRelevantClausesResult:
        """Return relevant clauses and structured conditions for the calculation backend.

        No financial calculations are performed here — only retrieval,
        categorization, and structuring of policy rules.
        """
        # Local import to avoid a hard import cycle at module load time.
        from .policy_extractor import load_policy_details

        try:
            details = load_policy_details(policy_id)
        except PolicyNotFoundError:
            raise
        except Exception as exc:  # corrupt details file
            logger.warning("Could not load policy details for %s: %s", policy_id, exc)
            details = None

        query = self._treatment_query(treatment_details)
        chunks = self.retriever.retrieve(policy_id, query, top_k=10)

        clauses: List[ClaimRelevantClause] = []
        for chunk in chunks:
            clauses.append(
                ClaimRelevantClause(
                    clause_reference=chunk.clause_reference,
                    section_name=chunk.section_name,
                    page_number=chunk.page_number,
                    quoted_text=_excerpt(chunk.text),
                    condition_type=_classify_condition(chunk.text),
                    relevance_to_claim=(
                        "Retrieved clause may affect claim admissibility or payout."
                    ),
                )
            )

        structured_conditions = self._structured_conditions(details) if details else {}
        missing_information = (
            list(details.unknown_fields) if details and details.unknown_fields else []
        )

        return ClaimRelevantClausesResult(
            policy_id=policy_id,
            treatment_details=treatment_details,
            clauses=clauses,
            structured_conditions=structured_conditions,
            missing_information=missing_information,
        )

    @staticmethod
    def _treatment_query(treatment_details: Optional[dict]) -> str:
        if not treatment_details:
            return "coverage exclusions waiting period sub-limit claim condition"
        parts: List[str] = []
        for key, value in treatment_details.items():
            if value:
                parts.append(f"{key}: {value}")
        return " ".join(parts) if parts else "coverage exclusions waiting period"

    @staticmethod
    def _structured_conditions(details) -> dict:
        return {
            "insurer_name": details.insurer_name,
            "policy_name": details.policy_name,
            "sum_insured": details.sum_insured.model_dump() if details.sum_insured else None,
            "room_rent_limit": details.room_rent_limit.model_dump() if details.room_rent_limit else None,
            "icu_room_rent_limit": details.icu_room_rent_limit.model_dump() if details.icu_room_rent_limit else None,
            "copay_percentage": details.copay_percentage,
            "deductibles": details.deductibles,
            "waiting_periods": [w.model_dump() for w in details.waiting_periods],
            "exclusions": [e.model_dump() for e in details.exclusions],
            "sub_limits": [s.model_dump() for s in details.sub_limits],
            "pre_hospitalization_coverage_period": details.pre_hospitalization_coverage_period,
            "post_hospitalization_coverage_period": details.post_hospitalization_coverage_period,
            "claim_conditions": [c.model_dump() for c in details.claim_conditions],
        }


def _classify_condition(text: str) -> str:
    lowered = text.lower()
    if "waiting period" in lowered:
        return "waiting_period"
    if "exclud" in lowered:
        return "exclusion"
    if "room rent" in lowered or "sub-limit" in lowered or "sublimit" in lowered:
        return "limit"
    if "copay" in lowered or "co-pay" in lowered:
        return "copay"
    if "deduct" in lowered:
        return "deductible"
    if "icu" in lowered or "intensive care" in lowered:
        return "coverage"
    return "coverage"
