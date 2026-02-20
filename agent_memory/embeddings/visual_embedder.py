"""Open-CLIP wrapper for visual/spatial embeddings.

Embeds scene descriptions (text) using CLIP to provide an independent
retrieval channel based on spatial/perceptual similarity.
"""

from __future__ import annotations

import logging

import open_clip
import torch

from agent_memory.config import MEMORY_CONFIG

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Lazy-loading wrapper around open_clip for scene description embeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or MEMORY_CONFIG["clip_model"]
        self._model = None
        self._tokenizer = None
        self._dimension: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading CLIP model: %s", self._model_name)
        self._model, _, _ = open_clip.create_model_and_transforms(
            self._model_name, pretrained="openai",
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model.eval()

    @property
    def dimension(self) -> int:
        self._load()
        if self._dimension is None:
            # Infer from a dummy forward pass
            dummy = self._tokenizer(["test"])
            with torch.no_grad():
                out = self._model.encode_text(dummy)
            self._dimension = out.shape[-1]
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a scene description text using CLIP. Returns list of floats."""
        self._load()
        tokens = self._tokenizer([text])
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def embed_to_bytes(self, text: str) -> bytes:
        """Embed and return raw bytes for storage in SQLite BLOB column."""
        import struct
        floats = self.embed(text)
        return struct.pack(f"{len(floats)}f", *floats)

    def bytes_to_vector(self, data: bytes) -> list[float]:
        """Convert raw bytes back to float list."""
        import struct
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))
