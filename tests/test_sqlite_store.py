"""Tests for the SQLite store."""

import pytest
import pytest_asyncio

from agent_memory.models import Memory, SaveDecision, CompactionResult
from agent_memory.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


def _make_memory(**kwargs) -> Memory:
    defaults = dict(
        id="mem-1",
        content="Test memory content",
        raw_log_id="raw-1",
        session_id="sess-1",
        turn=1,
        valence=0.5,
        arousal=0.3,
        surprise=0.1,
        salience=0.6,
        keywords=[("python", 0.9), ("testing", 0.7)],
    )
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.mark.asyncio
async def test_save_and_get_memory(store):
    mem = _make_memory()
    await store.save_memory(mem)

    result = await store.get_memory("mem-1")
    assert result is not None
    assert result.id == "mem-1"
    assert result.content == "Test memory content"
    assert result.valence == 0.5
    assert result.arousal == 0.3
    assert len(result.keywords) == 2
    assert ("python", 0.9) in result.keywords


@pytest.mark.asyncio
async def test_get_nonexistent_memory(store):
    result = await store.get_memory("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_memories(store):
    for i in range(5):
        await store.save_memory(_make_memory(id=f"mem-{i}"))

    results = await store.list_memories(limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_list_memories_by_tier(store):
    await store.save_memory(_make_memory(id="hot-1", tier="hot"))
    await store.save_memory(_make_memory(id="warm-1", tier="warm"))
    await store.save_memory(_make_memory(id="hot-2", tier="hot"))

    hot = await store.list_memories(tier="hot")
    assert len(hot) == 2
    assert all(m.tier == "hot" for m in hot)


@pytest.mark.asyncio
async def test_count_memories(store):
    for i in range(4):
        await store.save_memory(_make_memory(id=f"mem-{i}"))

    assert await store.count_memories() == 4
    assert await store.count_memories(tier="hot") == 4
    assert await store.count_memories(tier="warm") == 0


@pytest.mark.asyncio
async def test_keyword_search(store):
    await store.save_memory(_make_memory(
        id="m1", keywords=[("python", 0.9), ("async", 0.7)],
    ))
    await store.save_memory(_make_memory(
        id="m2", keywords=[("javascript", 0.8), ("async", 0.6)],
    ))
    await store.save_memory(_make_memory(
        id="m3", keywords=[("python", 0.5), ("sync", 0.9)],
    ))

    results = await store.search_by_keywords(["python"])
    ids = {m.id for m in results}
    assert "m1" in ids
    assert "m3" in ids

    results = await store.search_by_keywords(["async"])
    ids = {m.id for m in results}
    assert "m1" in ids
    assert "m2" in ids


@pytest.mark.asyncio
async def test_update_memory_access(store):
    await store.save_memory(_make_memory())
    await store.update_memory_access("mem-1", decay_score=0.8, access_count=3, last_accessed="2026-02-20T12:00:00Z")

    mem = await store.get_memory("mem-1")
    assert mem.decay_score == 0.8
    assert mem.access_count == 3


@pytest.mark.asyncio
async def test_update_memory_tier(store):
    await store.save_memory(_make_memory())
    await store.update_memory_tier("mem-1", "cold")

    mem = await store.get_memory("mem-1")
    assert mem.tier == "cold"


@pytest.mark.asyncio
async def test_log_access(store):
    await store.save_memory(_make_memory())
    await store.log_access("acc-1", "mem-1", "2026-02-20T12:00:00Z", "primary", "sess-1", "test query")

    async with store.db.execute(
        "SELECT * FROM memory_access_log WHERE id = ?", ("acc-1",)
    ) as cur:
        row = await cur.fetchone()
        assert row is not None
        assert dict(row)["memory_id"] == "mem-1"


@pytest.mark.asyncio
async def test_log_save_decision(store):
    dec = SaveDecision(
        id="dec-1", raw_log_id="raw-1", session_id="sess-1",
        turn=1, decision="save", reason="Important", confidence=0.9,
    )
    await store.log_save_decision(dec)

    async with store.db.execute(
        "SELECT * FROM save_decisions WHERE id = ?", ("dec-1",)
    ) as cur:
        row = await cur.fetchone()
        assert row is not None
        assert dict(row)["decision"] == "save"


@pytest.mark.asyncio
async def test_raw_log_index(store):
    await store.index_raw_log("raw-1", "sess-1", 1, "2026-02-20T12:00:00Z", "/path/file.jsonl", 0)

    ref = await store.get_raw_log_ref("raw-1")
    assert ref is not None
    assert ref["session_id"] == "sess-1"
    assert ref["file_path"] == "/path/file.jsonl"


@pytest.mark.asyncio
async def test_compaction_logging(store):
    result = CompactionResult(
        id="comp-1", trigger="manual",
        memories_reviewed=10, memories_merged=3, notes="test run",
    )
    await store.log_compaction_run(result)
    await store.log_compaction_merge("comp-1", ["m1", "m2", "m3"], "m-new")

    async with store.db.execute(
        "SELECT * FROM compaction_runs WHERE id = ?", ("comp-1",)
    ) as cur:
        row = await cur.fetchone()
        assert dict(row)["memories_merged"] == 3

    async with store.db.execute(
        "SELECT * FROM compaction_merges WHERE compaction_id = ?", ("comp-1",)
    ) as cur:
        row = await cur.fetchone()
        assert dict(row)["resulting_memory_id"] == "m-new"


@pytest.mark.asyncio
async def test_get_compaction_candidates(store):
    # High decay, high salience = not a candidate
    await store.save_memory(_make_memory(id="m1", decay_score=0.9, salience=0.8))
    # Low decay, low salience = good candidate
    await store.save_memory(_make_memory(id="m2", decay_score=0.1, salience=0.2))
    # Fast-pathed = excluded
    await store.save_memory(_make_memory(id="m3", decay_score=0.1, salience=0.1, fast_pathed=True))
    # High access count gen-0 = excluded
    await store.save_memory(_make_memory(id="m4", decay_score=0.1, salience=0.1, access_count=10))

    candidates = await store.get_compaction_candidates(threshold=0.7)
    ids = {m.id for m in candidates}
    assert "m2" in ids
    assert "m1" not in ids
    assert "m3" not in ids
    assert "m4" not in ids
