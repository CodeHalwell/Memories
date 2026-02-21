"""Tests for A2.2 — emotional modulation of decay scoring."""

from datetime import datetime, timedelta, timezone

from agent_memory.core.decay import compute_decay


def test_emotional_boost_slows_decay():
    """High arousal + surprise should result in higher decay score than neutral."""
    old = datetime.now(timezone.utc) - timedelta(days=14)

    neutral = compute_decay(old, access_count=2, arousal=0.0, surprise=0.0)
    emotional = compute_decay(old, access_count=2, arousal=0.9, surprise=0.8)

    assert emotional > neutral


def test_semantic_floor():
    """Semantic memories should never fully decay — floor at 0.3 recency."""
    very_old = datetime.now(timezone.utc) - timedelta(days=365)

    episodic = compute_decay(very_old, access_count=0, is_semantic=False)
    semantic = compute_decay(very_old, access_count=0, is_semantic=True)

    assert semantic > episodic
    # Semantic floor: 0.6 * 0.3 + 0.4 * 0 = 0.18 minimum
    assert semantic >= 0.18


def test_emotional_boost_range():
    """Emotional boost should be bounded — max 2x slowdown."""
    t = datetime.now(timezone.utc) - timedelta(days=7)

    # Max boost: arousal=1.0, surprise=1.0 → boost = 2.0
    max_emotional = compute_decay(t, access_count=1, arousal=1.0, surprise=1.0)
    no_emotion = compute_decay(t, access_count=1, arousal=0.0, surprise=0.0)

    assert max_emotional > no_emotion
    # The boosted version should not be more than ~2x the base
    assert max_emotional < 2.0 * no_emotion + 0.1  # small tolerance


def test_backward_compatible_signature():
    """Original signature (without emotional params) should still work."""
    t = datetime.now(timezone.utc) - timedelta(days=5)
    score = compute_decay(t, access_count=3)
    assert 0.0 <= score <= 2.0


def test_semantic_with_emotional_boost():
    """Semantic + emotional should combine both effects."""
    old = datetime.now(timezone.utc) - timedelta(days=60)

    base = compute_decay(old, access_count=1)
    boosted = compute_decay(old, access_count=1, arousal=0.8, surprise=0.7, is_semantic=True)

    assert boosted > base
