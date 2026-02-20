"""Tests for the decay scoring function."""

from datetime import datetime, timedelta, timezone

from agent_memory.core.decay import compute_decay


def test_recent_high_access():
    """Recently accessed, frequently used memory should have high decay score."""
    now = datetime.now(timezone.utc)
    score = compute_decay(now, access_count=20)
    assert score > 0.7


def test_old_low_access():
    """Old, rarely accessed memory should have low decay score."""
    old = datetime.now(timezone.utc) - timedelta(days=60)
    score = compute_decay(old, access_count=1)
    assert score < 0.3


def test_never_accessed():
    """Memory accessed once long ago should decay significantly."""
    old = datetime.now(timezone.utc) - timedelta(days=30)
    score = compute_decay(old, access_count=0)
    assert score < 0.2


def test_just_accessed():
    """Memory accessed just now should have recency close to 1.0."""
    now = datetime.now(timezone.utc)
    score = compute_decay(now, access_count=0)
    assert score >= 0.55  # recency=1.0 * 0.6 + freq=0 * 0.4 = 0.6


def test_recency_decay_curve():
    """Scores should decrease monotonically with time since access."""
    now = datetime.now(timezone.utc)
    scores = []
    for days in [0, 1, 7, 30, 90]:
        t = now - timedelta(days=days)
        scores.append(compute_decay(t, access_count=5))
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_frequency_boost():
    """Higher access count should produce higher score at same recency."""
    t = datetime.now(timezone.utc) - timedelta(days=10)
    low = compute_decay(t, access_count=1)
    high = compute_decay(t, access_count=50)
    assert high > low


def test_naive_datetime_handled():
    """Naive datetimes (no tzinfo) should not raise."""
    naive = datetime.now() - timedelta(days=5)
    score = compute_decay(naive, access_count=3)
    assert 0.0 <= score <= 2.0  # reasonable range
