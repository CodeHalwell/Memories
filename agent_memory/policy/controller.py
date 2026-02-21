"""Policy controller interface (A5 — stub for v2).

This module defines the interface for a learned memory policy that will
eventually replace fixed heuristics for save, retrieval, and compaction
decisions. For now, it delegates to the existing heuristic logic.

The hard constraints defined here cannot be overridden by any learned policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent_memory.config import MEMORY_CONFIG

logger = logging.getLogger(__name__)

POLICY_HARD_CONSTRAINTS = {
    # Save constraints
    "min_save_rate": 0.05,
    "max_save_rate": 0.50,
    "fast_path_override": True,

    # Retrieval constraints
    "min_layers": 1,
    "max_graph_depth": 4,
    "max_top_k": 20,

    # Compaction constraints
    "never_delete_raw_logs": True,
    "never_compact_fast_path_gen0": True,
    "require_merge_validation": True,
}


@dataclass
class PolicyState:
    """State vector for policy decisions (A5.2)."""
    turn_number: int = 0
    session_length: int = 0
    time_since_last_save: float = 0.0
    content_length: int = 0
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    emotional_surprise: float = 0.0
    hot_tier_count: int = 0
    recent_retrieval_hit_rate: float = 0.0
    retrieval_gap_score: float = 0.0
    graph_node_count: int = 0
    avg_edge_degree: float = 0.0
    orphan_memory_count: int = 0
    days_since_last_compaction: float = 0.0
    pending_merge_candidates: int = 0


class PolicyController:
    """Stub policy controller — uses heuristics, logs decisions for future training.

    In v2, this will be replaced with a learned model (gradient-boosted trees
    or a small RL-trained policy network).
    """

    def __init__(self) -> None:
        self.constraints = POLICY_HARD_CONSTRAINTS
        self.config = MEMORY_CONFIG

    def should_save(self, state: PolicyState, llm_confidence: float) -> bool:
        """Decide whether to save (heuristic — v1)."""
        return llm_confidence >= self.config["save_confidence_threshold"]

    def retrieval_config(self, state: PolicyState) -> dict:
        """Return retrieval parameters (heuristic — v1)."""
        return {
            "layers": self.config["retrieval_layers"],
            "graph_depth": min(
                self.config["graph_traversal_depth"],
                self.constraints["max_graph_depth"],
            ),
            "mood_weight": self.config["mood_congruent_weight"],
            "top_k": min(
                self.config["top_k_per_layer"],
                self.constraints["max_top_k"],
            ),
        }

    def compaction_priority(self, state: PolicyState) -> float:
        """Return compaction urgency score (heuristic — v1)."""
        if state.hot_tier_count > self.config["hot_tier_threshold"]:
            return 1.0
        return state.hot_tier_count / max(self.config["hot_tier_threshold"], 1)
