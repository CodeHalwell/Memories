"""Tests for retrieval gap-aware save decision (A2.1)."""

from agent_memory.core.save_decision import _compute_gap_overlap


def test_gap_overlap_full():
    content_kw = ["python", "async", "web"]
    gap_kw = ["python", "async", "web"]
    assert _compute_gap_overlap(content_kw, gap_kw) == 1.0


def test_gap_overlap_partial():
    content_kw = ["python", "async", "web", "api"]
    gap_kw = ["python", "web"]
    overlap = _compute_gap_overlap(content_kw, gap_kw)
    assert overlap == 0.5  # 2/4


def test_gap_overlap_none():
    content_kw = ["python", "async"]
    gap_kw = ["javascript", "react"]
    assert _compute_gap_overlap(content_kw, gap_kw) == 0.0


def test_gap_overlap_empty_content():
    assert _compute_gap_overlap([], ["python"]) == 0.0


def test_gap_overlap_empty_gaps():
    assert _compute_gap_overlap(["python"], []) == 0.0


def test_gap_overlap_both_empty():
    assert _compute_gap_overlap([], []) == 0.0
