"""Feature 4 (part 1): Embeddings via Sentence Transformers.

Uses sentence-transformers/all-MiniLM-L6-v2 by default — small enough to run
locally on a normal laptop, and the model is cached on disk after download.
"""

from __future__ import annotations

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .exceptions import EmbeddingError
from .utils import EMBEDDING_MODEL, logger

_embedding_model: Optional["EmbeddingModel"] = None


class EmbeddingModel:
    """Wrapper around a Sentence Transformer model."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None

    def _load(self) -> None:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # network or model-file errors
                raise EmbeddingError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns [] for empty input."""
        if not texts:
            return []
        self._load()
        try:
            vectors = self._model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_texts([text])[0]


def get_embedding_model() -> EmbeddingModel:
    """Module-level singleton embedding model (lazy load)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
