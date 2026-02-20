"""Sentence-transformers wrapper for text embeddings.

Used for semantic similarity search in the Qdrant vector store.
Runs locally on CPU or GPU.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from agent_memory.config import MEMORY_CONFIG

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Lazy-loading wrapper around sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or MEMORY_CONFIG["text_embedding_model"]
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading text embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns a list of float lists."""
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
