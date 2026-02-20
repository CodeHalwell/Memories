"""Graph-informed keyword reweighting (A2.5).

Keywords that appear in memories connected by RELATES_TO edges are
structurally more important than ones in isolated memories. This module
runs a lightweight reweighting pass during compaction, boosting weights
for keywords shared across well-connected memory clusters.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


async def reweight_keywords_from_graph(
    sqlite: SQLiteStore,
    graph: GraphStore,
    max_hops: int = 2,
) -> int:
    """Adjust keyword weights based on graph connectivity.

    Keywords shared across well-connected memories get boosted.
    Returns the number of keyword weights updated.
    """
    # Get all keywords with their memory associations
    rows = await sqlite.get_all_keywords_with_memories()
    if not rows:
        return 0

    # Group by keyword
    keyword_index: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        keyword_index[row["keyword"]].append(row)

    updated = 0
    for keyword, entries in keyword_index.items():
        if len(entries) < 2:
            continue

        memory_ids = [e["memory_id"] for e in entries]

        # Check graph connectivity between memories sharing this keyword
        connected_pairs = 0
        total_pairs = 0
        for i, m1 in enumerate(memory_ids):
            for m2 in memory_ids[i + 1:]:
                total_pairs += 1
                if graph.path_exists(m1, m2, max_hops=max_hops):
                    connected_pairs += 1

        if total_pairs == 0:
            continue

        connectivity_ratio = connected_pairs / total_pairs

        # Boost weight for highly connected keywords
        # Scale: 0.0 connectivity = no change, 1.0 = +50% weight
        if connectivity_ratio <= 0.0:
            continue

        boost = 1.0 + 0.5 * connectivity_ratio

        for entry in entries:
            new_weight = min(entry["weight"] * boost, 1.0)
            if new_weight != entry["weight"]:
                await sqlite.update_keyword_weight(
                    entry["memory_id"], keyword, new_weight,
                )
                updated += 1

    logger.info("Keyword reweighting: updated %d weights", updated)
    return updated
