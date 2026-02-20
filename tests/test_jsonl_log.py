"""Tests for the JSONL logger and raw log indexer."""

import json
from pathlib import Path

import pytest

from agent_memory.models import RawLogEntry
from agent_memory.storage.jsonl_log import JSONLLogger


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def logger(log_dir):
    return JSONLLogger(log_dir)


def test_append_creates_file(logger, log_dir):
    entry = RawLogEntry(
        id="entry-1",
        session_id="sess-1",
        turn=1,
        content="Hello world",
        role="assistant",
    )
    file_path, offset = logger.append(entry)
    assert Path(file_path).exists()
    assert offset == 0
    assert "sess-1.jsonl" in file_path


def test_append_increments_offset(logger):
    entry1 = RawLogEntry(id="e1", session_id="s1", turn=1, content="First")
    entry2 = RawLogEntry(id="e2", session_id="s1", turn=2, content="Second")

    _, offset1 = logger.append(entry1)
    _, offset2 = logger.append(entry2)

    assert offset1 == 0
    assert offset2 > 0


def test_read_entry_at_offset(logger):
    entry = RawLogEntry(id="e1", session_id="s1", turn=1, content="Test content")
    file_path, offset = logger.append(entry)

    recovered = logger.read_entry(file_path, offset)
    assert recovered.id == "e1"
    assert recovered.content == "Test content"
    assert recovered.turn == 1


def test_read_entry_multiple(logger):
    entries = [
        RawLogEntry(id=f"e{i}", session_id="s1", turn=i, content=f"Content {i}")
        for i in range(5)
    ]
    refs = [logger.append(e) for e in entries]

    for i, (fp, off) in enumerate(refs):
        recovered = logger.read_entry(fp, off)
        assert recovered.id == f"e{i}"
        assert recovered.content == f"Content {i}"


def test_iter_session(logger):
    for i in range(3):
        logger.append(RawLogEntry(id=f"e{i}", session_id="s1", turn=i, content=f"Msg {i}"))

    results = list(logger.iter_session("s1"))
    assert len(results) == 3
    assert results[0].id == "e0"
    assert results[2].id == "e2"


def test_iter_empty_session(logger):
    results = list(logger.iter_session("nonexistent"))
    assert results == []


def test_search(logger):
    logger.append(RawLogEntry(id="e1", session_id="s1", turn=1, content="The cat sat on the mat"))
    logger.append(RawLogEntry(id="e2", session_id="s1", turn=2, content="The dog ran in the park"))
    logger.append(RawLogEntry(id="e3", session_id="s1", turn=3, content="A cat and a dog"))

    results = logger.search("s1", "cat")
    assert len(results) == 2
    ids = {r.id for r in results}
    assert "e1" in ids
    assert "e3" in ids


def test_list_sessions(logger):
    logger.append(RawLogEntry(id="e1", session_id="alpha", turn=1, content="a"))
    logger.append(RawLogEntry(id="e2", session_id="beta", turn=1, content="b"))

    sessions = logger.list_sessions()
    assert set(sessions) == {"alpha", "beta"}


def test_session_size(logger):
    assert logger.session_size("nonexistent") == 0
    logger.append(RawLogEntry(id="e1", session_id="s1", turn=1, content="data"))
    assert logger.session_size("s1") > 0


def test_separate_session_files(logger, log_dir):
    logger.append(RawLogEntry(id="e1", session_id="s1", turn=1, content="a"))
    logger.append(RawLogEntry(id="e2", session_id="s2", turn=1, content="b"))

    assert (log_dir / "s1.jsonl").exists()
    assert (log_dir / "s2.jsonl").exists()
