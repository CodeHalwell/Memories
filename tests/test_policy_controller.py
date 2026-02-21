"""Tests for the policy controller stub (A5)."""

from agent_memory.policy.controller import (
    POLICY_HARD_CONSTRAINTS,
    PolicyController,
    PolicyState,
)


def test_policy_state_defaults():
    state = PolicyState()
    assert state.turn_number == 0
    assert state.hot_tier_count == 0
    assert state.recent_retrieval_hit_rate == 0.0


def test_policy_controller_save_heuristic():
    ctrl = PolicyController()
    state = PolicyState()

    assert ctrl.should_save(state, llm_confidence=0.8) is True
    assert ctrl.should_save(state, llm_confidence=0.3) is False


def test_policy_controller_retrieval_config():
    ctrl = PolicyController()
    state = PolicyState()
    config = ctrl.retrieval_config(state)

    assert "layers" in config
    assert "graph_depth" in config
    assert config["graph_depth"] <= POLICY_HARD_CONSTRAINTS["max_graph_depth"]
    assert config["top_k"] <= POLICY_HARD_CONSTRAINTS["max_top_k"]


def test_policy_controller_compaction_priority():
    ctrl = PolicyController()

    # Below threshold
    state = PolicyState(hot_tier_count=100)
    priority = ctrl.compaction_priority(state)
    assert 0.0 < priority < 1.0

    # Above threshold
    state = PolicyState(hot_tier_count=600)
    priority = ctrl.compaction_priority(state)
    assert priority == 1.0


def test_hard_constraints():
    assert POLICY_HARD_CONSTRAINTS["never_delete_raw_logs"] is True
    assert POLICY_HARD_CONSTRAINTS["never_compact_fast_path_gen0"] is True
    assert POLICY_HARD_CONSTRAINTS["min_save_rate"] < POLICY_HARD_CONSTRAINTS["max_save_rate"]
