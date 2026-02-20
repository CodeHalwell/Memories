"""Tests for the data models."""

from agent_memory.models import Memory, RawLogEntry, SaveDecision, CompactionResult


def test_raw_log_entry_defaults():
    entry = RawLogEntry(content="test")
    assert entry.id  # should have UUID
    assert entry.timestamp  # should have timestamp
    assert entry.role == "assistant"
    assert entry.content == "test"


def test_memory_defaults():
    mem = Memory(content="test memory", raw_log_id="r1", session_id="s1", turn=1)
    assert mem.id
    assert mem.tier == "hot"
    assert mem.salience == 0.5
    assert mem.decay_score == 1.0
    assert mem.compaction_gen == 0
    assert mem.fast_pathed is False
    assert mem.is_semantic is False
    assert mem.keywords == []


def test_save_decision_defaults():
    dec = SaveDecision(raw_log_id="r1", session_id="s1", turn=1)
    assert dec.decision == "skip"
    assert dec.confidence == 0.0


def test_compaction_result_defaults():
    result = CompactionResult()
    assert result.trigger == "scheduled"
    assert result.memories_reviewed == 0
    assert result.memories_merged == 0
