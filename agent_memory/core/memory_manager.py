"""Memory Manager — orchestrates save, retrieve, and compact operations.

This is the primary interface for the agent runtime. It coordinates all
storage backends, embedding models, and the LLM-driven save decision pipeline.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.config import DATA_DIR, MEMORY_CONFIG
from agent_memory.core.compaction import CompactionEngine
from agent_memory.core.decay import compute_decay
from agent_memory.core.retrieval import RetrievalEngine
from agent_memory.core.save_decision import make_save_decision
from agent_memory.embeddings.text_embedder import TextEmbedder
from agent_memory.embeddings.visual_embedder import VisualEmbedder
from agent_memory.llm.client import llm_complete_json
from agent_memory.models import CompactionResult, Memory, RawLogEntry
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.jsonl_log import JSONLLogger
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_SCENE_SYSTEM = """You generate abstract, impressionistic scene descriptions for memories.
Not photorealistic — spatial and relational structure is the goal.

Respond with ONLY the scene description text. Example:
"A sparse room with two opposing ideas at opposite walls, connected by a fragile thread.
The atmosphere is tense, slightly dark. The more important concept occupies the centre and is larger."

Keep it to 2-3 sentences maximum."""


class MemoryManager:
    """Top-level orchestrator for the agent memory system."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.config = MEMORY_CONFIG

        # Storage backends
        self.jsonl = JSONLLogger(self.data_dir / "logs" / "sessions")
        self.sqlite = SQLiteStore(self.data_dir / "memory.db")
        self.graph = GraphStore(self.data_dir / "graph")
        self.vector = VectorStore(self.data_dir / "vectors")

        # Embedding models (lazy-loaded)
        self.text_embedder = TextEmbedder()
        self.visual_embedder = VisualEmbedder()

        # Sub-engines (initialized after storage is ready)
        self._retrieval: RetrievalEngine | None = None
        self._compaction: CompactionEngine | None = None

        # Track first turns per session
        self._first_turns: set[str] = set()

    async def initialize(self) -> None:
        """Initialize all storage backends. Must be called before any operations."""
        await self.sqlite.initialize()
        self.graph.initialize()
        self.vector.initialize(
            text_dim=self.text_embedder.dimension,
            visual_dim=self.visual_embedder.dimension,
        )

        self._retrieval = RetrievalEngine(
            sqlite=self.sqlite,
            graph=self.graph,
            vector=self.vector,
            text_embedder=self.text_embedder,
            visual_embedder=self.visual_embedder,
        )
        self._compaction = CompactionEngine(
            sqlite=self.sqlite,
            graph=self.graph,
            vector=self.vector,
            text_embedder=self.text_embedder,
            visual_embedder=self.visual_embedder,
        )

    async def initialize_lite(self) -> None:
        """Initialize without loading embedding models (for testing or lightweight use)."""
        await self.sqlite.initialize()
        self.graph.initialize()

    async def close(self) -> None:
        """Close all storage backends."""
        await self.sqlite.close()
        self.graph.close()
        self.vector.close()

    # ── Core operations ──

    async def process_turn(
        self, session_id: str, turn: int, content: str,
        role: str = "assistant", token_count: int = 0,
        model: str = "", provider: str = "",
    ) -> Memory | None:
        """Log an agent output and decide whether to save it as a memory.

        This is the main entry point called at the end of each turn.

        Returns the Memory if one was created, else None.
        """
        # 1. Create and persist raw log entry
        entry = RawLogEntry(
            session_id=session_id,
            turn=turn,
            content=content,
            role=role,
            token_count=token_count,
            model=model,
            provider=provider,
        )
        file_path, byte_offset = self.jsonl.append(entry)

        # 2. Index the raw log entry in SQLite
        await self.sqlite.index_raw_log(
            entry_id=entry.id,
            session_id=session_id,
            turn=turn,
            timestamp=entry.timestamp,
            file_path=file_path,
            byte_offset=byte_offset,
        )

        # 3. Run save decision
        is_first = session_id not in self._first_turns
        if is_first:
            self._first_turns.add(session_id)

        decision, memory = await make_save_decision(entry, is_first_turn=is_first)

        # 4. Log the decision
        await self.sqlite.log_save_decision(decision)

        if memory is None:
            return None

        # 5. Save the memory
        await self.sqlite.save_memory(memory)

        # 6. Create graph node
        self.graph.add_memory_node(
            memory_id=memory.id,
            summary=memory.summary or "",
            tier=memory.tier,
            salience=memory.salience,
            valence=memory.valence,
            compaction_gen=memory.compaction_gen,
            created_at=memory.created_at,
        )
        memory.graph_node_id = memory.id
        await self.sqlite.update_memory_graph_ref(memory.id, memory.id)

        # 7. Create text embedding and store in Qdrant
        try:
            text_vector = self.text_embedder.embed(memory.content)
            point_id = self.vector.upsert_text_vector(
                memory_id=memory.id,
                vector=text_vector,
                tier=memory.tier,
                valence=memory.valence,
                arousal=memory.arousal,
                session_id=session_id,
                created_at=memory.created_at,
            )
            memory.vector_id = point_id
            await self.sqlite.update_memory_vector_ref(memory.id, point_id)
        except Exception:
            logger.exception("Failed to create text embedding for memory %s", memory.id)

        # 8. Visual layer — generate scene description and CLIP embedding for salient memories
        if memory.salience > self.config["visual_salience_threshold"]:
            await self._generate_visual_layer(memory)

        return memory

    async def retrieve(
        self, query: str, session_id: str | None = None,
        top_k: int | None = None,
    ) -> list[Memory]:
        """Run three-layer retrieval and return ranked memories."""
        if self._retrieval is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")
        return await self._retrieval.retrieve(
            query=query, session_id=session_id, top_k=top_k,
        )

    async def run_compaction(self, trigger: str = "scheduled") -> CompactionResult:
        """Run a compaction cycle."""
        if self._compaction is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")
        return await self._compaction.run(trigger=trigger)

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Fetch a single memory and log the access."""
        mem = await self.sqlite.get_memory(memory_id)
        if mem is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        mem.access_count += 1
        mem.last_accessed = now
        mem.decay_score = compute_decay(
            datetime.fromisoformat(now), mem.access_count,
        )

        await self.sqlite.log_access(
            access_id=str(uuid.uuid4()),
            memory_id=memory_id,
            accessed_at=now,
            access_type="primary",
        )
        await self.sqlite.update_memory_access(
            memory_id, mem.decay_score, mem.access_count, now,
        )
        return mem

    # ── Visual layer ──

    async def _generate_visual_layer(self, memory: Memory) -> None:
        """Generate a scene description and CLIP embedding for a memory."""
        try:
            # Generate scene description via LLM
            from agent_memory.llm.client import llm_complete
            scene = await llm_complete(
                f"Generate an abstract scene description for this memory:\n\n{memory.content}",
                system=_SCENE_SYSTEM,
            )
            memory.scene_description = scene.strip()

            # Create CLIP embedding
            spatial_bytes = self.visual_embedder.embed_to_bytes(memory.scene_description)
            memory.spatial_embedding = spatial_bytes

            # Store in Qdrant visual collection
            visual_vector = self.visual_embedder.embed(memory.scene_description)
            self.vector.upsert_visual_vector(
                memory_id=memory.id,
                vector=visual_vector,
                session_id=memory.session_id,
                created_at=memory.created_at,
            )

            # Update SQLite
            await self.sqlite.update_memory_visual(
                memory.id, memory.scene_description, spatial_bytes,
            )
        except Exception:
            logger.exception("Visual layer generation failed for memory %s", memory.id)
