"""Decision-outcome pairing for policy training data (A4.3).

Outcomes are assessed asynchronously — either at session end or during the
next compaction cycle. Save decisions are assessed by checking whether the
saved memory was ever retrieved. Retrieval decisions are assessed by checking
whether the agent re-queried the same topic shortly afterward.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent_memory.config import MEMORY_CONFIG
from agent_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


async def assess_save_outcomes(
    sqlite: SQLiteStore,
    lookback_days: int | None = None,
) -> int:
    """Assess whether saved memories turned out to be useful.

    A memory is considered useful if it was retrieved at least once within
    the lookback window. Returns the number of decisions assessed.
    """
    lookback_days = lookback_days or MEMORY_CONFIG["save_outcome_lookback_days"]
    now = datetime.now(timezone.utc).isoformat()

    unassessed = await sqlite.get_unassessed_save_decisions(lookback_days)
    updated = 0

    for row in unassessed:
        useful = bool(row.get("access_count") and row["access_count"] > 0)
        await sqlite.update_save_outcome(row["id"], useful, now)
        updated += 1

    logger.info("Save outcome assessment: assessed %d decisions", updated)
    return updated


async def assess_retrieval_outcomes(
    sqlite: SQLiteStore,
    followup_turns: int | None = None,
    keyword_overlap_threshold: float | None = None,
) -> int:
    """Assess whether retrievals were helpful.

    Heuristic: if the agent did not re-query the same topic within N turns,
    the retrieval was probably adequate. Returns the number assessed.
    """
    followup_turns = followup_turns or MEMORY_CONFIG["retrieval_outcome_followup_turns"]
    overlap_threshold = keyword_overlap_threshold or MEMORY_CONFIG["retrieval_outcome_keyword_overlap"]
    now = datetime.now(timezone.utc).isoformat()

    unassessed = await sqlite.get_unassessed_retrieval_decisions()
    updated = 0

    for row in unassessed:
        turn = row.get("turn")
        if turn is None:
            continue

        followups = await sqlite.get_retrieval_followups(
            row["session_id"], turn, window=followup_turns,
        )

        original_keywords = set(
            w.lower().strip() for w in row["query"].split() if len(w) > 2
        )
        re_queried = False

        for fu_query in followups:
            fu_keywords = set(
                w.lower().strip() for w in fu_query.split() if len(w) > 2
            )
            if original_keywords and fu_keywords:
                overlap = len(original_keywords & fu_keywords) / max(len(original_keywords), 1)
                if overlap > overlap_threshold:
                    re_queried = True
                    break

        helpful = not re_queried
        await sqlite.update_retrieval_outcome(row["id"], helpful, now)
        updated += 1

    logger.info("Retrieval outcome assessment: assessed %d decisions", updated)
    return updated
