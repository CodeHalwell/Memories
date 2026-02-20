"""Tests for keyword reweighting from graph structure (A2.5)."""

import pytest
import pytest_asyncio

from agent_memory.core.keyword_reweight import reweight_keywords_from_graph
from agent_memory.models import Memory
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def graph(tmp_path):
    g = GraphStore(tmp_path / "graph")
    g.initialize()
    yield g
    g.close()


def _mem(id: str, **kwargs) -> Memory:
    defaults = dict(content="test", raw_log_id="r1", session_id="s1", turn=1)
    defaults.update(kwargs)
    return Memory(id=id, **defaults)


@pytest.mark.asyncio
async def test_reweight_boosts_connected_keywords(store, graph):
    """Keywords shared across graph-connected memories should be boosted."""
    # Create two memories with shared keyword and a graph connection
    await store.save_memory(_mem("m1", keywords=[("python", 0.5), ("async", 0.6)]))
    await store.save_memory(_mem("m2", keywords=[("python", 0.5), ("web", 0.7)]))

    graph.add_memory_node("m1", summary="")
    graph.add_memory_node("m2", summary="")
    graph.add_relates_to("m1", "m2", weight=0.8)

    updated = await reweight_keywords_from_graph(store, graph)
    assert updated > 0

    # Check that 'python' keyword was boosted for both memories
    m1 = await store.get_memory("m1")
    python_weight = dict(m1.keywords).get("python", 0)
    assert python_weight > 0.5  # boosted from 0.5


@pytest.mark.asyncio
async def test_reweight_no_change_for_disconnected(store, graph):
    """Keywords in disconnected memories should not be boosted."""
    await store.save_memory(_mem("m1", keywords=[("python", 0.5)]))
    await store.save_memory(_mem("m2", keywords=[("python", 0.5)]))

    graph.add_memory_node("m1", summary="")
    graph.add_memory_node("m2", summary="")
    # No edge between them

    updated = await reweight_keywords_from_graph(store, graph)
    assert updated == 0


@pytest.mark.asyncio
async def test_reweight_single_keyword_memory_skipped(store, graph):
    """Keywords appearing in only one memory are skipped."""
    await store.save_memory(_mem("m1", keywords=[("unique_concept", 0.5)]))

    graph.add_memory_node("m1", summary="")

    updated = await reweight_keywords_from_graph(store, graph)
    assert updated == 0


@pytest.mark.asyncio
async def test_reweight_caps_at_one(store, graph):
    """Boosted weights should never exceed 1.0."""
    await store.save_memory(_mem("m1", keywords=[("python", 0.95)]))
    await store.save_memory(_mem("m2", keywords=[("python", 0.95)]))

    graph.add_memory_node("m1", summary="")
    graph.add_memory_node("m2", summary="")
    graph.add_relates_to("m1", "m2", weight=0.9)

    await reweight_keywords_from_graph(store, graph)

    m1 = await store.get_memory("m1")
    python_weight = dict(m1.keywords).get("python", 0)
    assert python_weight <= 1.0
