"""Data models for the Agent Memory System."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class RawLogEntry:
    id: str = field(default_factory=_uuid)
    session_id: str = ""
    turn: int = 0
    timestamp: str = field(default_factory=_now)
    role: str = "assistant"
    content: str = ""
    token_count: int = 0
    model: str = ""
    provider: str = ""


@dataclass
class Memory:
    id: str = field(default_factory=_uuid)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    content: str = ""
    summary: str | None = None
    raw_log_id: str = ""
    session_id: str = ""
    turn: int = 0

    # Emotional metadata
    valence: float = 0.0
    arousal: float = 0.0
    surprise: float = 0.0

    # Salience and access
    salience: float = 0.5
    access_count: int = 0
    last_accessed: str | None = None
    decay_score: float = 1.0

    # Compaction state
    compaction_gen: int = 0
    tier: str = "hot"
    fast_pathed: bool = False
    is_semantic: bool = False

    # Cross-store references
    graph_node_id: str | None = None
    vector_id: str | None = None

    # Visual layer
    spatial_embedding: bytes | None = None
    scene_description: str | None = None

    # Keywords (not stored in main table — separate table)
    keywords: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class SaveDecision:
    id: str = field(default_factory=_uuid)
    raw_log_id: str = ""
    session_id: str = ""
    turn: int = 0
    decided_at: str = field(default_factory=_now)
    decision: str = "skip"  # save | skip | fast_path
    reason: str | None = None
    confidence: float = 0.0


@dataclass
class CompactionResult:
    id: str = field(default_factory=_uuid)
    ran_at: str = field(default_factory=_now)
    trigger: str = "scheduled"
    memories_reviewed: int = 0
    memories_merged: int = 0
    memories_pruned: int = 0
    notes: str | None = None
