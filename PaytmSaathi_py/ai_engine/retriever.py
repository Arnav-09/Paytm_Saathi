"""Feature 5: Semantic retrieval.

Accepts a natural-language question, retrieves the most relevant policy
clauses for a given policy_id, and returns them with provenance metadata and
relevance scores. Irrelevant chunks are dropped rather than returned to fill
top_k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .embeddings import get_embedding_model
from .exceptions import EmbeddingError
from .utils import MIN_RELEVANCE, TOP_K, logger
from .vector_store import ChromaVectorStore


@dataclass
class RetrievedChunk:
    """One retrieved chunk with provenance and a relevance score in [0, 1]."""

    chunk_id: str
    text: str
    page_number: int
    section_name: Optional[str]
    clause_reference: Optional[str]
    source_filename: str
    policy_id: str
    score: float


class PolicyRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedding_model=None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedding_model = embedding_model or get_embedding_model()

    def retrieve(
        self,
        policy_id: str,
        question: str,
        top_k: int = TOP_K,
        min_relevance: float = MIN_RELEVANCE,
    ) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a question within one policy.

        - Fetches more candidates than top_k, applies a keyword-overlap boost
          so exact policy wording is prioritized, then filters by
          min_relevance. Fewer than top_k chunks are returned when nothing
          else clears the threshold.
        """
        if not question or not question.strip():
            return []

        try:
            query_embedding = self.embedding_model.embed_query(question)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed the question: {exc}") from exc

        # Ask for extra candidates so the relevance filter can drop weak ones.
        raw_hits = self.vector_store.query(
            policy_id,
            query_embedding,
            top_k=max(top_k * 3, 10),
            min_relevance=0.0,
        )

        question_terms = set(question.lower().split())
        scored: List[RetrievedChunk] = []
        for hit in raw_hits:
            score = hit["score"]
            # Keyword boost: exact policy terms matter more than pure semantics.
            chunk_terms = set(hit["text"].lower().split())
            overlap = len(question_terms & chunk_terms)
            score = min(1.0, score + 0.05 * overlap)
            if score < min_relevance:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=hit["chunk_id"],
                    text=hit["text"],
                    page_number=hit["page_number"],
                    section_name=hit["section_name"],
                    clause_reference=hit["clause_reference"],
                    source_filename=hit["source_filename"],
                    policy_id=policy_id,
                    score=score,
                )
            )

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        logger.info(
            "Retrieved %d chunks for policy %s (question=%r)",
            len(scored),
            policy_id,
            question[:80],
        )
        return scored[:top_k]
