"""Exploratory graph walks during sleep (A3).

During the compaction sleep cycle, performs semi-random walks through the
memory graph and vector space to discover non-obvious connections between
memories encoded separately. This mimics REM-like stochastic association
discovery that prevents overfitting to mundane patterns.

Two strategies:
  1. Random anchor pairs — sample pairs from different sessions, check
     semantic similarity, classify relationship via LLM.
  2. Cluster bridges — find memories close in vector space but disconnected
     in the graph. These are "latent" connections.
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_complete
from agent_memory.models import DiscoveredEdge
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_CLASSIFY_SYSTEM = """Classify the relationship between two memories.

Respond with exactly one word from this list:
caused, supports, contradicts, precedes, part_of, analogous, unrelated

If the connection is weak or speculative, respond "unrelated"."""


async def classify_relationship(
    mem_a_content: str, mem_b_content: str,
) -> str:
    """Ask the LLM to classify the relationship between two memories."""
    prompt = (
        f"Memory A: {mem_a_content}\n\nMemory B: {mem_b_content}\n\n"
        "What is the relationship?"
    )
    try:
        response = await llm_complete(prompt, system=_CLASSIFY_SYSTEM, temperature=0.1)
        result = response.strip().lower()
        valid = {"caused", "supports", "contradicts", "precedes", "part_of", "analogous", "unrelated"}
        return result if result in valid else "unrelated"
    except Exception:
        logger.debug("Relationship classification failed")
        return "unrelated"


async def exploratory_walk(
    sqlite: SQLiteStore,
    graph: GraphStore,
    vector: VectorStore,
    text_embedder=None,
    n_walks: int | None = None,
    similarity_threshold: float | None = None,
    max_new_edges: int | None = None,
) -> list[DiscoveredEdge]:
    """Perform semi-random walks to discover non-obvious memory connections.

    Returns a list of discovered edges. Caller is responsible for committing
    them to the graph.
    """
    cfg = MEMORY_CONFIG
    n_walks = n_walks or cfg["dream_walk_count"]
    similarity_threshold = similarity_threshold or cfg["dream_similarity_threshold"]
    max_new_edges = max_new_edges or cfg["dream_max_new_edges"]

    discovered: list[DiscoveredEdge] = []

    # Get all memories with vector embeddings
    all_memories = await sqlite.get_memories_with_vectors(tiers=["hot", "warm"])
    if len(all_memories) < 2:
        return discovered

    # Strategy 1: Random anchor pairs
    if not text_embedder:
        return discovered

    for _ in range(n_walks):
        if len(discovered) >= max_new_edges:
            break

        a, b = random.sample(all_memories, 2)

        # Skip if same session (likely already connected)
        if a["session_id"] == b["session_id"]:
            continue

        # Skip if already connected in graph
        if graph.path_exists(a["id"], b["id"], max_hops=1):
            continue

        # Check semantic similarity via vector store
        sim = vector.similarity(a["vector_id"], b["vector_id"])
        if sim is None or sim < similarity_threshold:
            continue

        # Load full memories for classification
        mem_a = await sqlite.get_memory(a["id"])
        mem_b = await sqlite.get_memory(b["id"])
        if not mem_a or not mem_b:
            continue

        rel_type = await classify_relationship(mem_a.content, mem_b.content)
        if rel_type != "unrelated":
            discovered.append(DiscoveredEdge(
                source_id=a["id"],
                target_id=b["id"],
                similarity=sim,
                relationship_type=rel_type,
                discovery_method="random_walk",
            ))

    return discovered[:max_new_edges]


async def commit_discoveries(
    discoveries: list[DiscoveredEdge],
    graph: GraphStore,
    sqlite: SQLiteStore,
    run_id: str | None = None,
) -> int:
    """Commit discovered edges to the graph and log the exploration run.

    Returns the number of edges committed.
    """
    now = datetime.now(timezone.utc).isoformat()
    run_id = run_id or str(uuid.uuid4())
    strategies = list({d.discovery_method for d in discoveries})
    committed = 0

    for edge in discoveries:
        edge_id = str(uuid.uuid4())
        try:
            graph.add_relates_to(
                from_id=edge.source_id,
                to_id=edge.target_id,
                weight=edge.similarity,
                relationship_type=edge.relationship_type,
                created_at=now,
            )
            committed += 1
            await sqlite.log_dream_edge(
                edge_id=edge_id,
                run_id=run_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                similarity=edge.similarity,
                relationship_type=edge.relationship_type,
                discovery_method=edge.discovery_method,
                committed=True,
            )
        except Exception:
            logger.debug("Failed to commit edge %s -> %s", edge.source_id, edge.target_id)
            await sqlite.log_dream_edge(
                edge_id=edge_id,
                run_id=run_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                similarity=edge.similarity,
                relationship_type=edge.relationship_type,
                discovery_method=edge.discovery_method,
                committed=False,
            )

    await sqlite.log_dream_run(
        run_id=run_id,
        ran_at=now,
        n_walks=len(discoveries),
        edges_discovered=len(discoveries),
        edges_committed=committed,
        strategies=strategies,
        notes=f"Committed {committed}/{len(discoveries)} edges",
    )

    logger.info("Dream exploration: committed %d/%d edges", committed, len(discoveries))
    return committed
