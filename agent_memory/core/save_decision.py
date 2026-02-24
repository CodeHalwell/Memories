"""LLM-driven save decision and keyword extraction.

At the end of each turn, this module determines whether the output should be
persisted as a memory. The LLM scores emotional dimensions, extracts keywords,
and provides a confidence-weighted save/skip decision.

A2.1 Amendment: the save decision is informed by retrieval gaps — if recent
queries failed to find relevant memories in a topic area, the save threshold
for overlapping content is lowered temporarily.
"""

from __future__ import annotations

import logging

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_complete_json
from agent_memory.models import Memory, RawLogEntry, SaveDecision
from agent_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


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


async def get_retrieval_gaps(sqlite: SQLiteStore, session_id: str) -> list[str]:
    """Identify topic areas where recent retrievals returned poor results (A2.1).

    Returns keywords from queries that yielded low-relevance results.
    """
    return await sqlite.get_failed_retrieval_keywords(
        session_id,
        lookback=MEMORY_CONFIG["gap_lookback_turns"],
    )


def _compute_gap_overlap(content_keywords: list[str], gap_keywords: list[str]) -> float:
    """Compute overlap between content keywords and retrieval gap keywords."""
    if not content_keywords or not gap_keywords:
        return 0.0
    content_set = set(content_keywords)
    gap_set = set(gap_keywords)
    return len(content_set & gap_set) / max(len(content_set), 1)


async def make_save_decision(
    entry: RawLogEntry,
    is_first_turn: bool = False,
    sqlite: SQLiteStore | None = None,
) -> tuple[SaveDecision, Memory | None]:
    """Decide whether to save an agent output as a memory.

    Args:
        entry: The raw log entry to evaluate.
        is_first_turn: Whether this is the first turn of the session.
        sqlite: SQLite store for retrieval gap awareness (A2.1). Optional.

    Returns:
        (SaveDecision, Memory or None) — the decision log entry, and a Memory
        if the decision is to save.
    """
    cfg = MEMORY_CONFIG

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

Session: {entry.session_id}
Turn: {entry.turn}
Content:
<content>
{entry.content}
</content>

Respond with JSON only."""

    try:
        result = await llm_complete_json(prompt, system=cfg["prompts"]["save_decision"])
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

    # Extract keywords for gap analysis
    keywords = [
        (kw["keyword"].lower(), float(kw.get("weight", 1.0)))
        for kw in result.get("keywords", [])
    ][:cfg["max_keywords_per_memory"]]
    content_kw_names = [kw for kw, _ in keywords]

    # A2.1: Retrieval gap awareness — lower threshold if content fills a gap
    threshold = cfg["save_confidence_threshold"]
    gap_triggered = False
    if sqlite:
        try:
            gap_keywords = await get_retrieval_gaps(sqlite, entry.session_id)
            gap_overlap = _compute_gap_overlap(content_kw_names, gap_keywords)
            if gap_overlap > cfg["gap_overlap_threshold"]:
                threshold *= cfg["gap_threshold_reduction"]
                gap_triggered = True
        except Exception:
            logger.debug("Gap detection failed, using default threshold")

    # Check fast path conditions from LLM-scored emotions
    fast_path = is_fast_path(arousal, surprise, entry.content)

    if fast_path:
        decision = "fast_path"
        should_save = True
        confidence = max(confidence, 0.9)
    elif should_save and confidence >= threshold:
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
        gap_triggered=gap_triggered,
        threshold_used=threshold,
    )

    if decision in ("save", "fast_path"):
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
