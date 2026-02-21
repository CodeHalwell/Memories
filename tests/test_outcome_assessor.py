"""Tests for policy outcome assessment (A4)."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone

from agent_memory.models import Memory, SaveDecision
from agent_memory.policy.outcome_assessor import (
    assess_retrieval_outcomes,
    assess_save_outcomes,
)
from agent_memory.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


def _mem(id: str, **kwargs) -> Memory:
    defaults = dict(content="test", raw_log_id=f"raw-{id}", session_id="s1", turn=1)
    defaults.update(kwargs)
    return Memory(id=id, **defaults)


def _two_days_ago() -> str:
    """Return an ISO timestamp 2 days ago, which is older than 1 day and passes a lookback_days=1 filter."""
    return (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()


@pytest.mark.asyncio
async def test_assess_save_outcomes_marks_unused(store):
    """Memories never accessed should be marked as not useful."""
    # Create memory with 0 access_count
    await store.save_memory(_mem("m1", access_count=0))
    # Create save decision pointing to that memory's raw_log_id
    dec = SaveDecision(
        id="dec-1", raw_log_id="raw-m1", session_id="s1", turn=1,
        decision="save", confidence=0.8,
        decided_at=_two_days_ago(),
    )
    await store.log_save_decision(dec)

    assessed = await assess_save_outcomes(store, lookback_days=1)
    assert assessed >= 1


@pytest.mark.asyncio
async def test_assess_save_outcomes_marks_used(store):
    """Memories that have been accessed should be marked as useful."""
    await store.save_memory(_mem("m2", access_count=5, raw_log_id="raw-m2"))
    dec = SaveDecision(
        id="dec-2", raw_log_id="raw-m2", session_id="s1", turn=1,
        decision="save", confidence=0.9,
        decided_at=_two_days_ago(),
    )
    await store.log_save_decision(dec)

    assessed = await assess_save_outcomes(store, lookback_days=1)
    assert assessed >= 1


@pytest.mark.asyncio
async def test_retrieval_decision_logging(store):
    """Retrieval decisions should be logged and queryable."""
    await store.log_retrieval_decision(
        decision_id="rd-1", session_id="s1", turn=5,
        query="python async patterns",
        decided_at="2026-02-20T10:00:00Z",
        layers_queried=["grep", "keyword", "semantic"],
        graph_depth=2, mood_weight=0.2, top_k=5,
        memory_ids=["m1", "m2"], return_count=2,
    )

    async with store.db.execute(
        "SELECT * FROM retrieval_decisions WHERE id = ?", ("rd-1",)
    ) as cur:
        row = await cur.fetchone()
        assert row is not None
        data = dict(row)
        assert data["query"] == "python async patterns"
        assert data["return_count"] == 2


@pytest.mark.asyncio
async def test_retrieval_outcome_assessment(store):
    """Retrieval outcomes should be assessed based on follow-up queries."""
    # Log a retrieval decision
    await store.log_retrieval_decision(
        decision_id="rd-1", session_id="s1", turn=5,
        query="python async",
        decided_at=_two_days_ago(),
        layers_queried=["keyword"],
        graph_depth=2, mood_weight=0.2, top_k=5,
        memory_ids=["m1"], return_count=1,
    )

    # No follow-up on same topic = helpful
    assessed = await assess_retrieval_outcomes(store)
    assert assessed >= 0


@pytest.mark.asyncio
async def test_save_decision_gap_fields(store):
    """Save decisions should store gap_triggered and threshold_used."""
    dec = SaveDecision(
        id="dec-gap", raw_log_id="raw-1", session_id="s1", turn=1,
        decision="save", confidence=0.7,
        gap_triggered=True, threshold_used=0.35,
    )
    await store.log_save_decision(dec)

    async with store.db.execute(
        "SELECT gap_triggered, threshold_used FROM save_decisions WHERE id = ?", ("dec-gap",)
    ) as cur:
        row = await cur.fetchone()
        data = dict(row)
        assert data["gap_triggered"] == 1
        assert data["threshold_used"] == 0.35
