"""SQLite storage for memory metadata, access tracking, and compaction history."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from agent_memory.config import DB_PATH
from agent_memory.models import CompactionResult, Memory, SaveDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_log_index (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    byte_offset INTEGER
);

CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    content           TEXT NOT NULL,
    summary           TEXT,
    raw_log_id        TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    turn              INTEGER NOT NULL,
    valence           REAL,
    arousal           REAL,
    surprise          REAL,
    salience          REAL DEFAULT 0.5,
    access_count      INTEGER DEFAULT 0,
    last_accessed     TEXT,
    decay_score       REAL DEFAULT 1.0,
    compaction_gen    INTEGER DEFAULT 0,
    tier              TEXT DEFAULT 'hot',
    fast_pathed       INTEGER DEFAULT 0,
    is_semantic       INTEGER DEFAULT 0,
    graph_node_id     TEXT,
    vector_id         TEXT,
    spatial_embedding BLOB,
    scene_description TEXT,
    FOREIGN KEY (raw_log_id) REFERENCES raw_log_index(id)
);

CREATE TABLE IF NOT EXISTS memory_keywords (
    memory_id   TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (memory_id, keyword),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_keyword ON memory_keywords(keyword);

CREATE TABLE IF NOT EXISTS memory_access_log (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_type TEXT NOT NULL,
    session_id  TEXT,
    query       TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS compaction_runs (
    id                  TEXT PRIMARY KEY,
    ran_at              TEXT NOT NULL,
    trigger             TEXT,
    memories_reviewed   INTEGER,
    memories_merged     INTEGER,
    memories_pruned     INTEGER,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS compaction_merges (
    compaction_id         TEXT NOT NULL,
    source_memory_ids     TEXT NOT NULL,
    resulting_memory_id   TEXT NOT NULL,
    FOREIGN KEY (compaction_id) REFERENCES compaction_runs(id)
);

CREATE TABLE IF NOT EXISTS save_decisions (
    id          TEXT PRIMARY KEY,
    raw_log_id  TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    decided_at  TEXT NOT NULL,
    decision    TEXT NOT NULL,
    reason      TEXT,
    confidence  REAL
);
"""


class SQLiteStore:
    """Async SQLite store for memory metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "SQLiteStore not initialized — call initialize() first"
        return self._db

    # ── Raw log index ──

    async def index_raw_log(
        self, entry_id: str, session_id: str, turn: int,
        timestamp: str, file_path: str, byte_offset: int,
    ) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO raw_log_index (id, session_id, turn, timestamp, file_path, byte_offset) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry_id, session_id, turn, timestamp, file_path, byte_offset),
        )
        await self.db.commit()

    async def get_raw_log_ref(self, entry_id: str) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM raw_log_index WHERE id = ?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Memories ──

    async def save_memory(self, mem: Memory) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO memories
            (id, created_at, updated_at, content, summary, raw_log_id, session_id, turn,
             valence, arousal, surprise, salience, access_count, last_accessed, decay_score,
             compaction_gen, tier, fast_pathed, is_semantic, graph_node_id, vector_id,
             spatial_embedding, scene_description)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mem.id, mem.created_at, mem.updated_at, mem.content, mem.summary,
                mem.raw_log_id, mem.session_id, mem.turn,
                mem.valence, mem.arousal, mem.surprise, mem.salience,
                mem.access_count, mem.last_accessed, mem.decay_score,
                mem.compaction_gen, mem.tier, int(mem.fast_pathed), int(mem.is_semantic),
                mem.graph_node_id, mem.vector_id,
                mem.spatial_embedding, mem.scene_description,
            ),
        )
        # Save keywords
        await self.db.execute("DELETE FROM memory_keywords WHERE memory_id = ?", (mem.id,))
        for kw, weight in mem.keywords:
            await self.db.execute(
                "INSERT INTO memory_keywords (memory_id, keyword, weight) VALUES (?, ?, ?)",
                (mem.id, kw, weight),
            )
        await self.db.commit()

    async def get_memory(self, memory_id: str) -> Memory | None:
        async with self.db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            mem = _row_to_memory(dict(row))

        # Load keywords
        async with self.db.execute(
            "SELECT keyword, weight FROM memory_keywords WHERE memory_id = ?", (memory_id,)
        ) as cur:
            mem.keywords = [(r["keyword"], r["weight"]) async for r in cur]
        return mem

    async def list_memories(
        self, tier: str | None = None, limit: int = 100, offset: int = 0,
    ) -> list[Memory]:
        if tier:
            sql = "SELECT * FROM memories WHERE tier = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params: tuple = (tier, limit, offset)
        else:
            sql = "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params = (limit, offset)

        memories = []
        async with self.db.execute(sql, params) as cur:
            async for row in cur:
                memories.append(_row_to_memory(dict(row)))

        # Batch-load keywords
        if memories:
            ids = [m.id for m in memories]
            placeholders = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({placeholders})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in memories:
                m.keywords = kw_map.get(m.id, [])

        return memories

    async def count_memories(self, tier: str | None = None) -> int:
        if tier:
            sql = "SELECT COUNT(*) as cnt FROM memories WHERE tier = ?"
            params: tuple = (tier,)
        else:
            sql = "SELECT COUNT(*) as cnt FROM memories"
            params = ()
        async with self.db.execute(sql, params) as cur:
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def update_memory_access(
        self, memory_id: str, decay_score: float, access_count: int, last_accessed: str,
    ) -> None:
        await self.db.execute(
            "UPDATE memories SET decay_score = ?, access_count = ?, last_accessed = ?, updated_at = ? WHERE id = ?",
            (decay_score, access_count, last_accessed, last_accessed, memory_id),
        )
        await self.db.commit()

    async def update_memory_tier(self, memory_id: str, tier: str) -> None:
        await self.db.execute(
            "UPDATE memories SET tier = ? WHERE id = ?", (tier, memory_id)
        )
        await self.db.commit()

    async def update_memory_graph_ref(self, memory_id: str, graph_node_id: str) -> None:
        await self.db.execute(
            "UPDATE memories SET graph_node_id = ? WHERE id = ?", (graph_node_id, memory_id)
        )
        await self.db.commit()

    async def update_memory_vector_ref(self, memory_id: str, vector_id: str) -> None:
        await self.db.execute(
            "UPDATE memories SET vector_id = ? WHERE id = ?", (vector_id, memory_id)
        )
        await self.db.commit()

    async def update_memory_visual(
        self, memory_id: str, scene_description: str, spatial_embedding: bytes,
    ) -> None:
        await self.db.execute(
            "UPDATE memories SET scene_description = ?, spatial_embedding = ? WHERE id = ?",
            (scene_description, spatial_embedding, memory_id),
        )
        await self.db.commit()

    # ── Keyword search ──

    async def search_by_keywords(
        self, keywords: list[str], limit: int = 10,
    ) -> list[Memory]:
        if not keywords:
            return []
        placeholders = ",".join("?" for _ in keywords)
        sql = f"""
            SELECT m.*, SUM(mk.weight) as match_score
            FROM memories m
            JOIN memory_keywords mk ON m.id = mk.memory_id
            WHERE mk.keyword IN ({placeholders})
            GROUP BY m.id
            ORDER BY match_score * m.decay_score DESC
            LIMIT ?
        """
        memories = []
        async with self.db.execute(sql, (*keywords, limit)) as cur:
            async for row in cur:
                memories.append(_row_to_memory(dict(row)))

        # Load keywords for results
        if memories:
            ids = [m.id for m in memories]
            ph = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in memories:
                m.keywords = kw_map.get(m.id, [])

        return memories

    # ── Access log ──

    async def log_access(
        self, access_id: str, memory_id: str, accessed_at: str,
        access_type: str, session_id: str | None = None, query: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO memory_access_log (id, memory_id, accessed_at, access_type, session_id, query) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (access_id, memory_id, accessed_at, access_type, session_id, query),
        )
        await self.db.commit()

    # ── Save decisions ──

    async def log_save_decision(self, dec: SaveDecision) -> None:
        await self.db.execute(
            "INSERT INTO save_decisions (id, raw_log_id, session_id, turn, decided_at, decision, reason, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (dec.id, dec.raw_log_id, dec.session_id, dec.turn, dec.decided_at, dec.decision, dec.reason, dec.confidence),
        )
        await self.db.commit()

    # ── Compaction ──

    async def log_compaction_run(self, result: CompactionResult) -> None:
        await self.db.execute(
            "INSERT INTO compaction_runs (id, ran_at, trigger, memories_reviewed, memories_merged, memories_pruned, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (result.id, result.ran_at, result.trigger, result.memories_reviewed, result.memories_merged, result.memories_pruned, result.notes),
        )
        await self.db.commit()

    async def log_compaction_merge(
        self, compaction_id: str, source_ids: list[str], resulting_id: str,
    ) -> None:
        await self.db.execute(
            "INSERT INTO compaction_merges (compaction_id, source_memory_ids, resulting_memory_id) VALUES (?, ?, ?)",
            (compaction_id, json.dumps(source_ids), resulting_id),
        )
        await self.db.commit()

    async def get_compaction_candidates(self, threshold: float = 0.7) -> list[Memory]:
        """Return memories that are candidates for compaction."""
        sql = """
            SELECT m.* FROM memories m
            LEFT JOIN (
                SELECT memory_id, COUNT(*) as edge_count
                FROM memory_keywords
                GROUP BY memory_id
            ) kc ON m.id = kc.memory_id
            WHERE m.tier = 'hot'
              AND m.fast_pathed = 0
              AND NOT (m.compaction_gen = 0 AND m.access_count > 5)
            ORDER BY ((1 - m.decay_score) * 0.6 + (1 - m.salience) * 0.4) DESC
        """
        candidates = []
        async with self.db.execute(sql) as cur:
            async for row in cur:
                mem = _row_to_memory(dict(row))
                score = (1 - mem.decay_score) * 0.6 + (1 - mem.salience) * 0.4
                if score > threshold:
                    candidates.append(mem)

        # Load keywords
        if candidates:
            ids = [m.id for m in candidates]
            ph = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in candidates:
                m.keywords = kw_map.get(m.id, [])

        return candidates


def _row_to_memory(row: dict) -> Memory:
    """Convert a SQLite row dict to a Memory dataclass."""
    # Filter out extra columns like match_score that aren't Memory fields
    known = {f.name for f in Memory.__dataclass_fields__.values()} if hasattr(Memory, '__dataclass_fields__') else set()
    return Memory(
        id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        content=row["content"],
        summary=row.get("summary"),
        raw_log_id=row["raw_log_id"],
        session_id=row["session_id"],
        turn=row["turn"],
        valence=row.get("valence", 0.0),
        arousal=row.get("arousal", 0.0),
        surprise=row.get("surprise", 0.0),
        salience=row.get("salience", 0.5),
        access_count=row.get("access_count", 0),
        last_accessed=row.get("last_accessed"),
        decay_score=row.get("decay_score", 1.0),
        compaction_gen=row.get("compaction_gen", 0),
        tier=row.get("tier", "hot"),
        fast_pathed=bool(row.get("fast_pathed", 0)),
        is_semantic=bool(row.get("is_semantic", 0)),
        graph_node_id=row.get("graph_node_id"),
        vector_id=row.get("vector_id"),
        spatial_embedding=row.get("spatial_embedding"),
        scene_description=row.get("scene_description"),
    )
