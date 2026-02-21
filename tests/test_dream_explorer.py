"""Tests for dream exploration (A3)."""

import pytest
import pytest_asyncio

from agent_memory.core.dream_explorer import commit_discoveries
from agent_memory.models import DiscoveredEdge
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


@pytest.mark.asyncio
async def test_commit_discoveries(store, graph):
    """Discovered edges should be committed to the graph and logged."""
    graph.add_memory_node("m1", summary="Memory about Python")
    graph.add_memory_node("m2", summary="Memory about coding")

    discoveries = [
        DiscoveredEdge(
            source_id="m1",
            target_id="m2",
            similarity=0.85,
            relationship_type="supports",
            discovery_method="random_walk",
        )
    ]

    committed = await commit_discoveries(discoveries, graph, store)
    assert committed == 1

    # Verify edge exists in graph
    related = graph.get_related_memories("m1", max_depth=1)
    assert any(r["id"] == "m2" for r in related)


@pytest.mark.asyncio
async def test_commit_empty_discoveries(store, graph):
    """Empty discovery list should commit nothing."""
    committed = await commit_discoveries([], graph, store)
    assert committed == 0


@pytest.mark.asyncio
async def test_commit_logs_dream_run(store, graph):
    """Dream runs should be logged in the database."""
    graph.add_memory_node("m1", summary="A")
    graph.add_memory_node("m2", summary="B")

    discoveries = [
        DiscoveredEdge(
            source_id="m1", target_id="m2",
            similarity=0.8, relationship_type="analogous",
            discovery_method="random_walk",
        )
    ]

    await commit_discoveries(discoveries, graph, store, run_id="test-run")

    async with store.db.execute(
        "SELECT * FROM dream_exploration_runs WHERE id = ?", ("test-run",)
    ) as cur:
        row = await cur.fetchone()
        assert row is not None
        assert dict(row)["edges_committed"] == 1


@pytest.mark.asyncio
async def test_commit_logs_individual_edges(store, graph):
    """Individual edges should be logged in dream_discovered_edges."""
    graph.add_memory_node("m1", summary="A")
    graph.add_memory_node("m2", summary="B")

    discoveries = [
        DiscoveredEdge(
            source_id="m1", target_id="m2",
            similarity=0.9, relationship_type="supports",
            discovery_method="cluster_bridge",
        )
    ]

    await commit_discoveries(discoveries, graph, store, run_id="test-run")

    async with store.db.execute(
        "SELECT * FROM dream_discovered_edges WHERE exploration_run_id = ?", ("test-run",)
    ) as cur:
        rows = [dict(r) async for r in cur]
        assert len(rows) == 1
        assert rows[0]["committed"] == 1
        assert rows[0]["relationship_type"] == "supports"
        assert rows[0]["discovery_method"] == "cluster_bridge"


@pytest.mark.asyncio
async def test_discovered_edge_model():
    """DiscoveredEdge dataclass should hold all expected fields."""
    edge = DiscoveredEdge(
        source_id="a", target_id="b",
        similarity=0.75, relationship_type="analogous",
        discovery_method="random_walk",
    )
    assert edge.source_id == "a"
    assert edge.similarity == 0.75
