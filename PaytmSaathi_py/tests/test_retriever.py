"""Tests for retriever.py (Feature 5) with mocked embeddings and vector store."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_engine.exceptions import EmbeddingError, PolicyNotFoundError
from ai_engine.retriever import PolicyRetriever, RetrievedChunk


class FakeEmbeddingModel:
    def embed_query(self, text: str):
        return [0.1, 0.2, 0.3]


def _make_store(hits):
    store = MagicMock()
    store.query.return_value = hits
    return store


def test_retrieve_filters_by_relevance_and_ranks():
    hits = [
        {
            "chunk_id": "c1",
            "text": "Room rent limit is 5000 per day.",
            "page_number": 12,
            "section_name": "Room Rent",
            "clause_reference": "4.2",
            "source_filename": "policy.pdf",
            "policy_id": "p1",
            "score": 0.9,
        },
        {
            "chunk_id": "c2",
            "text": "ICU treatment is covered.",
            "page_number": 15,
            "section_name": "ICU",
            "clause_reference": "5.1",
            "source_filename": "policy.pdf",
            "policy_id": "p1",
            "score": 0.5,
        },
        {
            "chunk_id": "c3",
            "text": "Unrelated greeting text.",
            "page_number": 1,
            "section_name": "Intro",
            "clause_reference": "1.1",
            "source_filename": "policy.pdf",
            "policy_id": "p1",
            "score": 0.1,
        },
    ]
    with patch("ai_engine.retriever.get_embedding_model", return_value=FakeEmbeddingModel()):
        retriever = PolicyRetriever(vector_store=_make_store(hits))
        results = retriever.retrieve("p1", "room rent limit", top_k=2, min_relevance=0.3)

    assert [r.chunk_id for r in results] == ["c1", "c2"]
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].page_number == 12
    assert results[0].section_name == "Room Rent"
    assert results[0].clause_reference == "4.2"
    # Irrelevant chunk (score 0.1 + keyword boost) must not fill top_k.
    assert all(r.score >= 0.3 for r in results)
    store_call = retriever.vector_store.query.call_args
    assert store_call.args[0] == "p1"
    assert store_call.kwargs["min_relevance"] == 0.0


def test_retrieve_returns_fewer_than_top_k_when_evidence_is_weak():
    hits = [
        {
            "chunk_id": "c1",
            "text": "Some generic text.",
            "page_number": 1,
            "section_name": None,
            "clause_reference": None,
            "source_filename": "policy.pdf",
            "policy_id": "p1",
            "score": 0.05,
        }
    ]
    with patch("ai_engine.retriever.get_embedding_model", return_value=FakeEmbeddingModel()):
        retriever = PolicyRetriever(vector_store=_make_store(hits))
        results = retriever.retrieve("p1", "room rent limit", top_k=5, min_relevance=0.5)

    assert results == []


def test_retrieve_empty_question_returns_empty():
    with patch("ai_engine.retriever.get_embedding_model", return_value=FakeEmbeddingModel()):
        retriever = PolicyRetriever(vector_store=_make_store([]))
        assert retriever.retrieve("p1", "   ") == []


def test_retrieve_unknown_policy_raises_policy_not_found():
    store = MagicMock()
    store.query.side_effect = PolicyNotFoundError("p9")
    with patch("ai_engine.retriever.get_embedding_model", return_value=FakeEmbeddingModel()):
        retriever = PolicyRetriever(vector_store=store)
        with pytest.raises(PolicyNotFoundError):
            retriever.retrieve("p9", "anything")


def test_retrieve_embedding_failure_raises_embedding_error():
    class BrokenEmbedding:
        def embed_query(self, text):
            raise RuntimeError("model down")

    store = MagicMock()
    with patch("ai_engine.retriever.get_embedding_model", return_value=BrokenEmbedding()):
        retriever = PolicyRetriever(vector_store=store)
        with pytest.raises(EmbeddingError):
            retriever.retrieve("p1", "room rent")
