"""Tests for compaction addendum features (A2.3, A2.4)."""

from agent_memory.core.compaction import _can_merge, _cosine_similarity
from agent_memory.config import MEMORY_CONFIG
from agent_memory.models import Memory, MergeValidation


def _mem(**kwargs) -> Memory:
    defaults = dict(content="test", raw_log_id="r1", session_id="s1", turn=1)
    defaults.update(kwargs)
    return Memory(**defaults)


def test_generation_gap_guard_blocks_merge():
    """A2.3: Memories with >1 generation gap should not merge."""
    group = [
        _mem(compaction_gen=0, valence=0.5),
        _mem(compaction_gen=3, valence=0.5),
    ]
    assert not _can_merge(group, MEMORY_CONFIG)


def test_generation_gap_guard_allows_adjacent():
    """A2.3: Memories within 1 generation should merge."""
    group = [
        _mem(compaction_gen=1, valence=0.5),
        _mem(compaction_gen=2, valence=0.5),
    ]
    assert _can_merge(group, MEMORY_CONFIG)


def test_generation_gap_guard_allows_same():
    """A2.3: Same-generation memories should merge."""
    group = [
        _mem(compaction_gen=2, valence=0.5),
        _mem(compaction_gen=2, valence=0.5),
    ]
    assert _can_merge(group, MEMORY_CONFIG)


def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(a, b) - 1.0) < 0.001


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 0.001


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(_cosine_similarity(a, b) + 1.0) < 0.001


def test_merge_validation_model():
    v = MergeValidation(
        passed=True,
        avg_source_score=0.85,
        avg_merged_score=0.80,
        degradation=0.05,
        queries_tested=["q1", "q2"],
    )
    assert v.passed
    assert v.degradation == 0.05
    assert len(v.queries_tested) == 2


def test_merge_validation_default():
    v = MergeValidation()
    assert v.passed is True
    assert v.queries_tested == []
