"""Feature 4 (part 2): ChromaDB vector store.

Persists embeddings locally under data/vector_db. Each policy gets its own
ChromaDB collection so documents from different policies stay isolated.
Duplicate ingestion of the same policy is prevented via a policy registry.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from .exceptions import PolicyNotFoundError, VectorStoreError
from .schemas import Chunk
from .utils import CHROMA_PERSIST_DIR, ensure_dir, logger, sanitize_collection_name

POLICY_REGISTRY_COLLECTION = "__policies_registry__"


class ChromaVectorStore:
    """Thin wrapper around ChromaDB's PersistentClient."""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        ensure_dir(self.persist_dir)
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc

    # ------------------------------------------------------------------
    # Collection naming
    # ------------------------------------------------------------------
    @staticmethod
    def _policy_collection_name(policy_id: str) -> str:
        return sanitize_collection_name(f"policy_{policy_id}")

    def add_chunks(
        self,
        policy_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add chunks for a policy. Returns the number of chunks added.

        Skips ingestion (returns 0) when the policy is already registered,
        preventing duplicate vectors when the same policy is uploaded again.
        """
        if not chunks:
            return 0
        if self.is_ingested(policy_id):
            logger.info("Policy %s already ingested; skipping duplicate.", policy_id)
            return 0

        collection = self.client.get_or_create_collection(
            name=self._policy_collection_name(policy_id)
        )
        try:
            metadatas = [
                {
                    "policy_id": policy_id,
                    "page_number": chunk.page_number,
                    "section_name": chunk.section_name or "",
                    "clause_reference": chunk.clause_reference or "",
                    "source_filename": chunk.source_filename or "",
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ]
            collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
                embeddings=embeddings,
            )
            self._registry().add(
                ids=[policy_id],
                documents=[policy_id],
                metadatas=[
                    {
                        "policy_id": policy_id,
                        "chunk_count": len(chunks),
                        "source_filename": chunks[0].source_filename or "",
                    }
                ],
            )
            logger.info("Stored %d chunks for policy %s", len(chunks), policy_id)
            return len(chunks)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to add chunks for policy {policy_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Deletion / isolation
    # ------------------------------------------------------------------
    def delete_policy(self, policy_id: str) -> bool:
        """Drop a policy's collection and registry entry. Returns True."""
        try:
            try:
                self.client.delete_collection(
                    name=self._policy_collection_name(policy_id)
                )
            except Exception:
                # Collection may not exist; that is fine.
                pass
            self._registry().delete(ids=[policy_id])
            logger.info("Deleted policy %s from vector store", policy_id)
            return True
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete policy {policy_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        policy_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks for a policy, filtered by that policy's collection.

        ChromaDB returns L2 distances by default; they are mapped to a
        relevance score in [0, 1] with score = 1 / (1 + distance). Chunks
        below min_relevance are excluded so irrelevant chunks are never
        returned just to fill top_k.
        """
        collection_name = self._policy_collection_name(policy_id)
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception as exc:
            raise PolicyNotFoundError(policy_id) from exc

        try:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"policy_id": policy_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Vector query failed for policy {policy_id}: {exc}"
            ) from exc

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            if score < min_relevance:
                continue
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "page_number": int(metadata.get("page_number", 0)),
                    "section_name": metadata.get("section_name") or None,
                    "clause_reference": metadata.get("clause_reference") or None,
                    "source_filename": metadata.get("source_filename") or "",
                    "policy_id": policy_id,
                    "score": score,
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Metadata lookups (used by explain and claim-clauses features)
    # ------------------------------------------------------------------
    def count(self, policy_id: str) -> int:
        try:
            collection = self.client.get_collection(
                name=self._policy_collection_name(policy_id)
            )
            return collection.count()
        except Exception:
            return 0

    def get_by_metadata(
        self, policy_id: str, where_filter: dict[str, Any], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Fetch chunks whose metadata exactly matches where_filter."""
        collection_name = self._policy_collection_name(policy_id)
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception as exc:
            raise PolicyNotFoundError(policy_id) from exc
        try:
            result = collection.get(
                where=where_filter,
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Metadata lookup failed for policy {policy_id}: {exc}"
            ) from exc

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        hits = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "page_number": int(metadata.get("page_number", 0)),
                    "section_name": metadata.get("section_name") or None,
                    "clause_reference": metadata.get("clause_reference") or None,
                    "source_filename": metadata.get("source_filename") or "",
                    "policy_id": policy_id,
                }
            )
        return hits

    def _registry(self):
        return self.client.get_or_create_collection(name=POLICY_REGISTRY_COLLECTION)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def is_ingested(self, policy_id: str) -> bool:
        """True if this policy was already added to the vector store."""
        try:
            result = self._registry().get(ids=[policy_id])
            return bool(result.get("ids"))
        except Exception:
            return False
