"""Tests for rag_engine.py (Feature 6) with mocked retriever and LLM."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_engine.exceptions import (
    InvalidQuestionError,
    LLMProviderError,
    PolicyNotFoundError,
)
from ai_engine.rag_engine import RAGEngine, INSUFFICIENT_EVIDENCE_ANSWER
from ai_engine.retriever import RetrievedChunk
from ai_engine.schemas import RAGResponse, SourceCitation


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.received_policy_id = None
        self.received_question = None

    def retrieve(self, policy_id, question, **kwargs):
        self.received_policy_id = policy_id
        self.received_question = question
        return self.chunks


class AnsweringLLM:
    """Mock LLM that cites sources and knows nothing outside the context."""

    def generate(self, prompt, system_prompt=None, **kwargs):
        if "room rent" in prompt.lower():
            return "Your room rent limit is stated on page 12. [Source 1]"
        return INSUFFICIENT_EVIDENCE_ANSWER

    def generate_structured(self, prompt, schema, **kwargs):
        return {}


class FailingLLM:
    def generate(self, prompt, system_prompt=None, **kwargs):
        raise LLMProviderError("provider down")

    def generate_structured(self, prompt, schema, **kwargs):
        raise LLMProviderError("provider down")


def _chunk(chunk_id, text, page, section, clause, score):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=page,
        section_name=section,
        clause_reference=clause,
        source_filename="policy.pdf",
        policy_id="p1",
        score=score,
    )


def test_qa_returns_validated_response_with_citations():
    chunks = [
        _chunk(
            "c1",
            "Room rent limit is 5000 per day.",
            12,
            "Room Rent",
            "4.2",
            0.9,
        )
    ]
    engine = RAGEngine(retriever=FakeRetriever(chunks), llm_provider=AnsweringLLM())
    response = engine.ask("p1", "What is my room rent limit?")

    assert isinstance(response, RAGResponse)
    assert response.policy_id == "p1"
    assert response.question == "What is my room rent limit?"
    assert response.insufficient_evidence is False
    assert response.confidence == "high"
    assert len(response.sources) == 1
    assert response.sources[0] == SourceCitation(
        page_number=12,
        section_name="Room Rent",
        clause_reference="4.2",
        quoted_text="Room rent limit is 5000 per day.",
    )


def test_qa_insufficient_evidence_when_no_chunks():
    engine = RAGEngine(retriever=FakeRetriever([]), llm_provider=AnsweringLLM())
    response = engine.ask("p1", "Anything at all?")

    assert response.insufficient_evidence is True
    assert response.sources == []
    assert response.confidence == "low"
    assert INSUFFICIENT_EVIDENCE_ANSWER in response.answer


def test_qa_invalid_question_raises():
    engine = RAGEngine(retriever=FakeRetriever([]), llm_provider=AnsweringLLM())
    with pytest.raises(InvalidQuestionError):
        engine.ask("p1", "")
    with pytest.raises(InvalidQuestionError):
        engine.ask("p1", "   ")


def test_qa_llm_failure_propagates():
    chunks = [_chunk("c1", "some text", 1, None, None, 0.8)]
    engine = RAGEngine(retriever=FakeRetriever(chunks), llm_provider=FailingLLM())
    with pytest.raises(LLMProviderError):
        engine.ask("p1", "room rent")


def test_follow_up_conversation_is_preserved():
    chunks = [_chunk("c1", "some clause text", 5, "Section 2", "2.1", 0.8)]
    retriever = FakeRetriever(chunks)
    engine = RAGEngine(retriever=retriever, llm_provider=AnsweringLLM())

    r1 = engine.ask("p1", "What is covered?", conversation_id="conv-1")
    assert r1.conversation_id == "conv-1"
    r2 = engine.ask("p1", "And the limit?", conversation_id="conv-1")
    assert r2.conversation_id == "conv-1"
    # Both turns should be in memory so the follow-up prompt includes history.
    assert len(engine._conversations["conv-1"]) == 2


def test_unknown_policy_propagates_from_retriever():
    class RaisingRetriever:
        def retrieve(self, policy_id, question, **kwargs):
            from ai_engine.exceptions import PolicyNotFoundError

            raise PolicyNotFoundError(policy_id)

    engine = RAGEngine(retriever=RaisingRetriever(), llm_provider=AnsweringLLM())
    with pytest.raises(PolicyNotFoundError):
        engine.ask("missing", "q")
