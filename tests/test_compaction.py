"""Tests for the compaction engine — scoring, grouping, and merge rules."""

from agent_memory.core.compaction import (
    _can_merge,
    _group_by_keywords,
    _keyword_overlap,
    compaction_score,
)
from agent_memory.config import MEMORY_CONFIG
from agent_memory.models import Memory


def _mem(**kwargs) -> Memory:
    defaults = dict(
        content="test", raw_log_id="r1", session_id="s1", turn=1,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


def test_compaction_score_low_priority():
    """High decay + high salience = low compaction score (keep it)."""
    m = _mem(decay_score=0.9, salience=0.8)
    assert compaction_score(m) < 0.3


def test_compaction_score_high_priority():
    """Low decay + low salience = high compaction score (compact it)."""
    m = _mem(decay_score=0.1, salience=0.2)
    assert compaction_score(m) > 0.7


def test_keyword_overlap_identical():
    a = _mem(keywords=[("python", 1.0), ("async", 0.8)])
    b = _mem(keywords=[("python", 0.9), ("async", 0.7)])
    assert _keyword_overlap(a, b) == 1.0


def test_keyword_overlap_partial():
    a = _mem(keywords=[("python", 1.0), ("async", 0.8)])
    b = _mem(keywords=[("python", 0.9), ("web", 0.7)])
    overlap = _keyword_overlap(a, b)
    assert 0.3 <= overlap <= 0.4  # 1/3


def test_keyword_overlap_none():
    a = _mem(keywords=[("python", 1.0)])
    b = _mem(keywords=[("javascript", 1.0)])
    assert _keyword_overlap(a, b) == 0.0


def test_keyword_overlap_empty():
    a = _mem(keywords=[])
    b = _mem(keywords=[("python", 1.0)])
    assert _keyword_overlap(a, b) == 0.0


def test_can_merge_basic():
    group = [
        _mem(valence=0.5, fast_pathed=False, compaction_gen=1),
        _mem(valence=0.3, fast_pathed=False, compaction_gen=1),
    ]
    assert _can_merge(group, MEMORY_CONFIG)


def test_cannot_merge_opposite_valence():
    group = [
        _mem(valence=0.8),
        _mem(valence=-0.5),
    ]
    assert not _can_merge(group, MEMORY_CONFIG)


def test_cannot_merge_fast_pathed_gen0():
    group = [
        _mem(fast_pathed=True, compaction_gen=0),
        _mem(fast_pathed=False, compaction_gen=1),
    ]
    assert not _can_merge(group, MEMORY_CONFIG)


def test_can_merge_fast_pathed_gen1():
    """Fast-pathed memories past gen 0 can be merged."""
    group = [
        _mem(fast_pathed=True, compaction_gen=1),
        _mem(fast_pathed=False, compaction_gen=1),
    ]
    assert _can_merge(group, MEMORY_CONFIG)


def test_group_by_keywords_basic():
    mems = [
        _mem(id="a", keywords=[("python", 1.0), ("async", 0.8), ("web", 0.6)]),
        _mem(id="b", keywords=[("python", 0.9), ("async", 0.7), ("api", 0.5)]),
        _mem(id="c", keywords=[("javascript", 1.0), ("react", 0.8)]),
    ]
    groups = _group_by_keywords(mems, threshold=0.4)
    # a and b share 2/4 keywords (50% Jaccard) — should group
    # c shares nothing — should be alone (but groups need >1 member)
    assert len(groups) >= 1
    grouped_ids = {m.id for g in groups for m in g}
    assert "a" in grouped_ids and "b" in grouped_ids


def test_group_by_keywords_no_overlap():
    mems = [
        _mem(id="a", keywords=[("python", 1.0)]),
        _mem(id="b", keywords=[("javascript", 1.0)]),
    ]
    groups = _group_by_keywords(mems, threshold=0.6)
    assert len(groups) == 0  # no groups meet threshold
