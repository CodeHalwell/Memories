"""Decay scoring for memory access patterns.

Implements time-based exponential decay combined with frequency-based
persistence. Mirrors the forgetting curve — memories that aren't accessed
gradually lose retrieval priority.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def compute_decay(last_accessed: datetime, access_count: int) -> float:
    """Compute a decay score between 0.0 and ~1.0.

    Higher scores indicate more "alive" memories. The score combines:
      - Recency: exponential decay based on days since last access
      - Frequency: logarithmic scaling of access count

    Args:
        last_accessed: When the memory was last retrieved.
        access_count: Total number of times the memory has been retrieved.

    Returns:
        A float decay score, typically in [0.0, 1.0].
    """
    now = datetime.now(timezone.utc)
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    days_since = max((now - last_accessed).total_seconds() / 86400, 0.0)
    recency = math.exp(-0.1 * days_since)
    frequency = math.log1p(access_count) / 10.0
    return round(0.6 * recency + 0.4 * frequency, 4)
