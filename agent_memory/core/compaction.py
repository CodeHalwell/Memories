"""Compaction scheduler and merge logic.

Compaction runs between sessions (the "sleep cycle"). It collapses episodic
detail into semantic generalisations, implementing intentional forgetting.

Candidate memories are scored, grouped by keyword overlap, and merged when
appropriate. Lineage is tracked via EVOLVED_FROM edges so any compacted
memory can be traced back to its originals.

A2.3: Generation gap guard — only merge memories within 1 generation of each other.
A2.4: Merge validation via generative replay — synthetic queries test that
      merged memory still surfaces for the same queries as the originals.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_complete_json
from agent_memory.models import CompactionResult, Memory, MergeValidation
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MERGE_SYSTEM = """You are a memory compaction agent. You merge multiple related episodic memories
into a single generalised semantic memory.

Respond with ONLY valid JSON:
{
  "content": "merged memory content — a generalised summary preserving key facts",
  "summary": "one sentence summary",
  "keywords": [{"keyword": "term", "weight": 0.0-1.0}],
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "salience": 0.0 to 1.0
}

Guidelines:
- Preserve factual information but collapse redundant detail
- The merged memory should be more abstract and general than the originals
- Keyword list should be the union of important keywords from all source memories
- Emotional scores should reflect the blended tone of the source memories
- Salience should be the maximum salience of any source memory"""

_SYNTHETIC_QUERY_SYSTEM = """Generate search queries that would retrieve the given memory content.

Respond with ONLY a JSON array of query strings:
["query 1", "query 2", ...]

Generate queries that are natural search terms a user might use to find this information.
Be specific to the content — generic queries are not useful."""


def compaction_score(memory: Memory) -> float:
    """Score a memory for compaction candidacy.

    Low decay + low salience = good candidate for compaction.
    """
    return (1 - memory.decay_score) * 0.6 + (1 - memory.salience) * 0.4


def _keyword_overlap(mem_a: Memory, mem_b: Memory) -> float:
    """Compute keyword overlap ratio between two memories."""
    kw_a = {kw for kw, _ in mem_a.keywords}
    kw_b = {kw for kw, _ in mem_b.keywords}
    if not kw_a or not kw_b:
        return 0.0
    intersection = kw_a & kw_b
    union = kw_a | kw_b
    return len(intersection) / len(union) if union else 0.0


def _can_merge(group: list[Memory], config: dict) -> bool:
    """Check if a group of memories can be merged (no exclusion conditions)."""
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            # Opposite valence exclusion
            if (a.valence * b.valence < 0 and
                    abs(a.valence - b.valence) > config["valence_merge_exclusion_delta"]):
                return False
            # Either is fast_pathed gen-0
            if (a.fast_pathed and a.compaction_gen == 0) or \
               (b.fast_pathed and b.compaction_gen == 0):
                return False
            # A2.3: Generation gap guard — only merge within 1 generation
            if abs(a.compaction_gen - b.compaction_gen) > config.get("max_generation_gap_for_merge", 1):
                return False
    return True


def _group_by_keywords(candidates: list[Memory], threshold: float) -> list[list[Memory]]:
    """Group memories by keyword overlap using greedy clustering."""
    if not candidates:
        return []

    used = set()
    groups: list[list[Memory]] = []

    for i, mem_a in enumerate(candidates):
        if i in used:
            continue
        group = [mem_a]
        used.add(i)
        for j, mem_b in enumerate(candidates):
            if j in used:
                continue
            # Check overlap with all current group members
            overlaps = [_keyword_overlap(g, mem_b) for g in group]
            if overlaps and min(overlaps) >= threshold:
                group.append(mem_b)
                used.add(j)
        if len(group) > 1:
            groups.append(group)

    return groups


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(dot / norm) if norm > 0 else 0.0


async def validate_merge(
    source_memories: list[Memory],
    candidate_content: str,
    text_embedder,
    n_queries: int = 5,
    degradation_tolerance: float = 0.15,
) -> MergeValidation:
    """Validate a merge via generative replay (A2.4).

    Generate synthetic queries from source memories, then test whether the
    candidate merge still retrieves well for those queries.
    """
    source_text = "\n---\n".join(m.content for m in source_memories)
    prompt = f"Generate {n_queries} search queries for this content:\n\n{source_text}"

    try:
        queries = await llm_complete_json(prompt, system=_SYNTHETIC_QUERY_SYSTEM)
        if not isinstance(queries, list):
            queries = list(queries.values()) if isinstance(queries, dict) else []
    except Exception:
        logger.debug("Synthetic query generation failed, skipping validation")
        return MergeValidation(passed=True, queries_tested=[])

    if not queries:
        return MergeValidation(passed=True, queries_tested=[])

    # Embed candidate and sources
    candidate_emb = text_embedder.embed(candidate_content)
    source_embs = [text_embedder.embed(m.content) for m in source_memories]

    merged_scores = []
    source_scores = []

    for query in queries:
        query_emb = text_embedder.embed(str(query))
        merged_scores.append(_cosine_similarity(query_emb, candidate_emb))
        best_source = max(_cosine_similarity(query_emb, se) for se in source_embs)
        source_scores.append(best_source)

    avg_merged = sum(merged_scores) / len(merged_scores) if merged_scores else 0.0
    avg_source = sum(source_scores) / len(source_scores) if source_scores else 0.0
    degradation = avg_source - avg_merged

    return MergeValidation(
        passed=degradation < degradation_tolerance,
        avg_source_score=avg_source,
        avg_merged_score=avg_merged,
        degradation=degradation,
        queries_tested=[str(q) for q in queries],
    )


class CompactionEngine:
    """Runs compaction cycles — merging low-value memories into generalisations."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        graph: GraphStore,
        vector: VectorStore,
        text_embedder=None,
        visual_embedder=None,
    ) -> None:
        self.sqlite = sqlite
        self.graph = graph
        self.vector = vector
        self.text_embedder = text_embedder
        self.visual_embedder = visual_embedder
        self.config = MEMORY_CONFIG

    async def run(self, trigger: str = "scheduled") -> CompactionResult:
        """Execute a full compaction cycle.

        Steps:
          1. Select candidates from hot tier
          2. Filter by hard exclusions (graph edge count)
          3. Group by keyword overlap
          4. Merge eligible groups (with A2.4 validation)
          5. Update tiers
          6. Log the run
        """
        result = CompactionResult(trigger=trigger)

        # Get candidates from SQLite
        candidates = await self.sqlite.get_compaction_candidates(
            threshold=self.config["compaction_candidate_threshold"],
        )
        result.memories_reviewed = len(candidates)

        if not candidates:
            logger.info("Compaction: no candidates found")
            await self.sqlite.log_compaction_run(result)
            return result

        # Filter by graph edge count (structurally important anchors)
        filtered = []
        for mem in candidates:
            if mem.graph_node_id:
                edge_count = self.graph.get_edge_count(mem.id)
                if edge_count > 3:
                    continue
            filtered.append(mem)

        # Group by keyword overlap
        groups = _group_by_keywords(
            filtered, self.config["keyword_overlap_merge_threshold"],
        )

        merged_count = 0
        for group in groups:
            if not _can_merge(group, self.config):
                continue
            new_mem = await self._merge_group(group, result.id)
            if new_mem:
                merged_count += 1

        result.memories_merged = merged_count

        # Tier promotion: move hot memories exceeding threshold to warm
        hot_count = await self.sqlite.count_memories(tier="hot")
        threshold = self.config["hot_tier_threshold"]
        if hot_count > threshold:
            await self._promote_tier(hot_count - threshold)

        result.notes = f"Reviewed {result.memories_reviewed}, merged into {merged_count} semantic memories"
        await self.sqlite.log_compaction_run(result)

        logger.info(
            "Compaction complete: reviewed=%d, merged=%d",
            result.memories_reviewed, merged_count,
        )
        return result

    async def _merge_group(self, group: list[Memory], compaction_id: str) -> Memory | None:
        """Merge a group of memories into a single semantic memory."""
        # Build prompt with all source memories
        sources = "\n\n---\n\n".join(
            f"Memory {i+1} (salience={m.salience}, valence={m.valence}):\n{m.content}"
            for i, m in enumerate(group)
        )
        prompt = f"""Merge these {len(group)} related memories into a single generalised memory:

{sources}

Respond with JSON only."""

        try:
            result = await llm_complete_json(prompt, system=_MERGE_SYSTEM)
        except Exception:
            logger.exception("LLM merge failed for group of %d memories", len(group))
            return None

        merged_content = result.get("content", "")

        # A2.4: Validate merge via generative replay
        validation = None
        if self.text_embedder:
            validation = await validate_merge(
                source_memories=group,
                candidate_content=merged_content,
                text_embedder=self.text_embedder,
                n_queries=self.config.get("merge_validation_queries", 5),
                degradation_tolerance=self.config.get("merge_degradation_tolerance", 0.15),
            )
            if not validation.passed:
                logger.info(
                    "Merge validation failed (degradation=%.3f), skipping group of %d memories",
                    validation.degradation, len(group),
                )
                return None

        now = datetime.now(timezone.utc).isoformat()
        new_id = str(uuid.uuid4())
        max_gen = max(m.compaction_gen for m in group)

        keywords = [
            (kw["keyword"].lower(), float(kw.get("weight", 1.0)))
            for kw in result.get("keywords", [])
        ][:self.config["max_keywords_per_memory"]]

        new_mem = Memory(
            id=new_id,
            created_at=now,
            updated_at=now,
            content=merged_content,
            summary=result.get("summary"),
            raw_log_id=group[0].raw_log_id,  # link to first source
            session_id=group[0].session_id,
            turn=group[0].turn,
            valence=float(result.get("valence", 0.0)),
            arousal=float(result.get("arousal", 0.0)),
            salience=float(result.get("salience", 0.5)),
            compaction_gen=max_gen + 1,
            tier="warm",
            is_semantic=True,
            keywords=keywords,
        )

        # Save to SQLite
        await self.sqlite.save_memory(new_mem)

        # Create graph node for new memory
        self.graph.add_memory_node(
            memory_id=new_id,
            summary=new_mem.summary or "",
            tier="warm",
            salience=new_mem.salience,
            valence=new_mem.valence,
            compaction_gen=new_mem.compaction_gen,
            created_at=now,
        )
        await self.sqlite.update_memory_graph_ref(new_id, new_id)

        # Create EVOLVED_FROM edges and replicate RELATES_TO edges
        source_ids = [m.id for m in group]
        for src in group:
            self.graph.add_evolved_from(new_id, src.id, compaction_id=compaction_id, created_at=now)

        self.graph.replicate_edges_to_new_node(source_ids, new_id)

        # Move source memories to cold tier
        for src in group:
            await self.sqlite.update_memory_tier(src.id, "cold")
            if src.graph_node_id:
                self.graph.update_memory_tier(src.id, "cold")

        # Create text embedding for new memory
        if self.text_embedder:
            vector = self.text_embedder.embed(new_mem.content)
            point_id = self.vector.upsert_text_vector(
                memory_id=new_id, vector=vector, tier="warm",
                valence=new_mem.valence, arousal=new_mem.arousal,
                session_id=new_mem.session_id, created_at=now,
            )
            await self.sqlite.update_memory_vector_ref(new_id, point_id)

        # Log the merge with validation data
        await self.sqlite.log_compaction_merge(
            compaction_id, source_ids, new_id,
            validation_passed=validation.passed if validation else None,
            avg_source_score=validation.avg_source_score if validation else None,
            avg_merged_score=validation.avg_merged_score if validation else None,
            degradation=validation.degradation if validation else None,
        )

        return new_mem

    async def _promote_tier(self, count: int) -> None:
        """Move the oldest/lowest-decay hot memories to warm tier."""
        async with self.sqlite.db.execute(
            "SELECT id, graph_node_id FROM memories WHERE tier = 'hot' "
            "ORDER BY decay_score ASC LIMIT ?",
            (count,),
        ) as cur:
            rows = [dict(r) async for r in cur]

        for row in rows:
            await self.sqlite.update_memory_tier(row["id"], "warm")
            if row.get("graph_node_id"):
                self.graph.update_memory_tier(row["id"], "warm")
