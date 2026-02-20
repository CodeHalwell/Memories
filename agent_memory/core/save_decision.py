"""LLM-driven save decision and keyword extraction.

At the end of each turn, this module determines whether the output should be
persisted as a memory. The LLM scores emotional dimensions, extracts keywords,
and provides a confidence-weighted save/skip decision.
"""

from __future__ import annotations

import logging

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_complete_json
from agent_memory.models import Memory, RawLogEntry, SaveDecision

logger = logging.getLogger(__name__)

_SAVE_DECISION_SYSTEM = """You are a memory curator for an AI agent. Your job is to decide whether
an agent's output is worth remembering as a distinct memory.

Respond with ONLY valid JSON, no other text. Use this exact schema:

{
  "should_save": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "keywords": [{"keyword": "lowercase_term", "weight": 0.0-1.0}],
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "surprise": 0.0 to 1.0,
  "summary": "one sentence summary",
  "salience": 0.0 to 1.0
}

Guidelines:
- Keywords should be lowercase, use underscores for compound concepts (e.g., reinforcement_learning)
- Extract up to 10 keywords, each with a weight indicating relevance
- Valence: -1.0 (very negative) to 1.0 (very positive)
- Arousal: 0.0 (calm/routine) to 1.0 (intense/urgent)
- Surprise: 0.0 (expected) to 1.0 (completely unexpected)
- Salience: overall importance/memorability from 0.0 to 1.0
- Save routine/repetitive outputs with low confidence
- Save novel insights, decisions, errors, corrections, or user preferences with high confidence"""


def is_fast_path(arousal: float, surprise: float, content: str) -> bool:
    """Check if a memory should bypass the LLM save decision."""
    cfg = MEMORY_CONFIG
    if arousal > cfg["fast_path_arousal"] and surprise > cfg["fast_path_surprise"]:
        return True
    # Explicit user instruction markers
    lower = content.lower()
    if any(phrase in lower for phrase in ["remember this", "don't forget", "save this", "keep in mind"]):
        return True
    return False


async def make_save_decision(
    entry: RawLogEntry,
    is_first_turn: bool = False,
) -> tuple[SaveDecision, Memory | None]:
    """Decide whether to save an agent output as a memory.

    Returns:
        (SaveDecision, Memory or None) — the decision log entry, and a Memory
        if the decision is to save.
    """
    # First turn of a session is always saved via fast path
    if is_first_turn:
        mem = Memory(
            content=entry.content,
            raw_log_id=entry.id,
            session_id=entry.session_id,
            turn=entry.turn,
            salience=0.7,
            fast_pathed=True,
        )
        dec = SaveDecision(
            raw_log_id=entry.id,
            session_id=entry.session_id,
            turn=entry.turn,
            decision="fast_path",
            reason="First turn of session — always saved",
            confidence=1.0,
        )
        return dec, mem

    # Ask LLM for structured evaluation
    prompt = f"""Evaluate whether this agent output should be saved as a memory:

---
Session: {entry.session_id}
Turn: {entry.turn}
Content:
{entry.content}
---

Respond with JSON only."""

    try:
        result = await llm_complete_json(prompt, system=_SAVE_DECISION_SYSTEM)
    except Exception:
        logger.exception("LLM save decision failed, defaulting to skip")
        dec = SaveDecision(
            raw_log_id=entry.id,
            session_id=entry.session_id,
            turn=entry.turn,
            decision="skip",
            reason="LLM evaluation failed",
            confidence=0.0,
        )
        return dec, None

    confidence = float(result.get("confidence", 0.0))
    should_save = result.get("should_save", False)
    valence = float(result.get("valence", 0.0))
    arousal = float(result.get("arousal", 0.0))
    surprise = float(result.get("surprise", 0.0))
    salience = float(result.get("salience", 0.5))

    # Check fast path conditions from LLM-scored emotions
    fast_path = is_fast_path(arousal, surprise, entry.content)

    if fast_path:
        decision = "fast_path"
        should_save = True
        confidence = max(confidence, 0.9)
    elif should_save and confidence >= MEMORY_CONFIG["save_confidence_threshold"]:
        decision = "save"
    else:
        decision = "skip"

    dec = SaveDecision(
        raw_log_id=entry.id,
        session_id=entry.session_id,
        turn=entry.turn,
        decision=decision,
        reason=result.get("reason", ""),
        confidence=confidence,
    )

    if decision in ("save", "fast_path"):
        keywords = [
            (kw["keyword"].lower(), float(kw.get("weight", 1.0)))
            for kw in result.get("keywords", [])
        ][:MEMORY_CONFIG["max_keywords_per_memory"]]

        mem = Memory(
            content=entry.content,
            summary=result.get("summary"),
            raw_log_id=entry.id,
            session_id=entry.session_id,
            turn=entry.turn,
            valence=valence,
            arousal=arousal,
            surprise=surprise,
            salience=salience,
            fast_pathed=fast_path,
            keywords=keywords,
        )
        return dec, mem

    return dec, None
