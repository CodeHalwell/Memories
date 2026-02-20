"""Decay scoring for memory access patterns.

Implements time-based exponential decay combined with frequency-based
persistence. Mirrors the forgetting curve — memories that aren't accessed
gradually lose retrieval priority.

A2.2 Amendment: emotional salience (arousal + surprise) slows decay, and
semantic (compacted) memories have a floor preventing full decay.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from agent_memory.config import MEMORY_CONFIG


def compute_decay(
    last_accessed: datetime,
    access_count: int,
    arousal: float = 0.0,
    surprise: float = 0.0,
    is_semantic: bool = False,
) -> float:
    """Compute a decay score between 0.0 and ~1.0.

    Higher scores indicate more "alive" memories. The score combines:
      - Recency: exponential decay based on days since last access
      - Frequency: logarithmic scaling of access count
      - Emotional boost: high arousal + surprise slows decay (A2.2)
      - Semantic floor: compacted memories never fully decay (A2.2)

    Args:
        last_accessed: When the memory was last retrieved.
        access_count: Total number of times the memory has been retrieved.
        arousal: Emotional arousal score (0.0 to 1.0).
        surprise: Emotional surprise score (0.0 to 1.0).
        is_semantic: True if this memory is a product of compaction.

    Returns:
        A float decay score, typically in [0.0, 1.0].
    """
    now = datetime.now(timezone.utc)
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    days_since = max((now - last_accessed).total_seconds() / 86400, 0.0)

    halflife = MEMORY_CONFIG["decay_halflife_days"]
    lambda_ = math.log(2) / halflife if halflife > 0 else 0.1

    # A2.2: Emotional memories decay more slowly
    # arousal + surprise in [0, 2], so boost is in [1.0, 2.0]
    emotional_boost = 1.0 + 0.5 * (arousal + surprise)
    recency = math.exp(-lambda_ * days_since / emotional_boost)

    frequency = math.log1p(access_count) / 10.0

    # A2.2: Semantic (compacted) memories have a flatter decay curve
    if is_semantic:
        recency = max(recency, 0.3)

    return round(
        MEMORY_CONFIG["decay_recency_weight"] * recency
        + MEMORY_CONFIG["decay_frequency_weight"] * frequency,
        4,
    )
