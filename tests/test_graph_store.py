"""Tests for the Kuzu graph store."""

import pytest

from agent_memory.storage.graph_store import GraphStore


@pytest.fixture
def graph(tmp_path):
    g = GraphStore(tmp_path / "graph")
    g.initialize()
    yield g
    g.close()


def test_add_memory_node(graph):
    graph.add_memory_node("m1", summary="Test memory", salience=0.8)
    # Should not raise on duplicate
    graph.add_memory_node("m1", summary="Updated memory", salience=0.9)


def test_add_entity_node(graph):
    graph.add_entity_node("e1", name="Python", entity_type="concept")
    graph.add_entity_node("e1", name="Python", entity_type="concept")  # idempotent


def test_relates_to(graph):
    graph.add_memory_node("m1", summary="First")
    graph.add_memory_node("m2", summary="Second")
    graph.add_relates_to("m1", "m2", weight=0.8, relationship_type="supports")

    related = graph.get_related_memories("m1", max_depth=1)
    assert len(related) == 1
    assert related[0]["id"] == "m2"


def test_relates_to_multi_hop(graph):
    graph.add_memory_node("m1", summary="First")
    graph.add_memory_node("m2", summary="Second")
    graph.add_memory_node("m3", summary="Third")
    graph.add_relates_to("m1", "m2", weight=0.9, relationship_type="precedes")
    graph.add_relates_to("m2", "m3", weight=0.7, relationship_type="precedes")

    # Depth 1 should find m2 only
    related_1 = graph.get_related_memories("m1", max_depth=1)
    assert {r["id"] for r in related_1} == {"m2"}

    # Depth 2 should find m2 and m3
    related_2 = graph.get_related_memories("m1", max_depth=2)
    assert {r["id"] for r in related_2} == {"m2", "m3"}


def test_mentions(graph):
    graph.add_memory_node("m1", summary="About Python")
    graph.add_entity_node("e1", name="Python", entity_type="concept")
    graph.add_mentions("m1", "e1", weight=0.9)

    entities = graph.get_memory_entities("m1")
    assert len(entities) == 1
    assert entities[0]["name"] == "Python"


def test_evolved_from(graph):
    graph.add_memory_node("m1", summary="Original 1")
    graph.add_memory_node("m2", summary="Original 2")
    graph.add_memory_node("m3", summary="Compacted", compaction_gen=1)
    graph.add_evolved_from("m3", "m1", compaction_id="c1")
    graph.add_evolved_from("m3", "m2", compaction_id="c1")

    lineage = graph.get_evolution_lineage("m3")
    ids = {r["id"] for r in lineage}
    assert "m1" in ids
    assert "m2" in ids


def test_edge_count(graph):
    graph.add_memory_node("m1", summary="Center")
    graph.add_memory_node("m2", summary="A")
    graph.add_memory_node("m3", summary="B")
    graph.add_memory_node("m4", summary="C")

    graph.add_relates_to("m1", "m2")
    graph.add_relates_to("m1", "m3")
    graph.add_relates_to("m4", "m1")

    assert graph.get_edge_count("m1") == 3


def test_replicate_edges(graph):
    # Set up source nodes with edges
    graph.add_memory_node("s1", summary="Source 1")
    graph.add_memory_node("s2", summary="Source 2")
    graph.add_memory_node("ext", summary="External")
    graph.add_memory_node("new", summary="Compacted")

    graph.add_relates_to("s1", "ext", weight=0.7, relationship_type="supports")
    graph.add_relates_to("ext", "s2", weight=0.5, relationship_type="precedes")

    graph.replicate_edges_to_new_node(["s1", "s2"], "new")

    # New node should have the external edges
    related = graph.get_related_memories("new", max_depth=1)
    assert any(r["id"] == "ext" for r in related)


def test_update_memory_tier(graph):
    graph.add_memory_node("m1", summary="Test", tier="hot")
    graph.update_memory_tier("m1", "cold")
    # Verify by querying
    result = graph.conn.execute(
        "MATCH (m:Memory {id: $id}) RETURN m.tier", {"id": "m1"}
    )
    assert result.has_next()
    assert result.get_next()[0] == "cold"
