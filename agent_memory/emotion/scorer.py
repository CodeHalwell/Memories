"""Emotional scoring via LLM.

Scores the current context for valence and arousal to support mood-congruent
retrieval. Can also re-score existing memories during compaction.
"""

from __future__ import annotations

import logging

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_complete_json

logger = logging.getLogger(__name__)


async def score_emotion(text: str) -> dict[str, float]:
    """Score the emotional dimensions of a text.

    Returns dict with keys: valence, arousal, surprise.
    """
    prompt = f"Score the emotional tone of this text:\n\n{text}"

    try:
        result = await llm_complete_json(prompt, system=MEMORY_CONFIG["prompts"]["emotion"])
        return {
            "valence": _clamp(float(result.get("valence", 0.0)), -1.0, 1.0),
            "arousal": _clamp(float(result.get("arousal", 0.0)), 0.0, 1.0),
            "surprise": _clamp(float(result.get("surprise", 0.0)), 0.0, 1.0),
        }
    except Exception:
        logger.exception("Emotion scoring failed, returning neutral")
        return {"valence": 0.0, "arousal": 0.0, "surprise": 0.0}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
