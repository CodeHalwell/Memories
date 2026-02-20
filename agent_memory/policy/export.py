"""Policy training data export (A4.4).

Exports decision-outcome pairs as JSONL files for offline policy model training.
Requires sufficient assessed data before export is meaningful.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent_memory.config import MEMORY_CONFIG, POLICY_DATA_DIR
from agent_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


async def export_policy_training_data(
    sqlite: SQLiteStore,
    output_dir: Path | None = None,
) -> dict:
    """Export decision-outcome pairs for offline policy model training.

    Returns dict with export metadata: example counts and file paths.
    """
    output_dir = output_dir or POLICY_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = await sqlite.export_save_policy_data()
    retrieval_data = await sqlite.export_retrieval_policy_data()

    save_path = output_dir / "save_policy_data.jsonl"
    retrieval_path = output_dir / "retrieval_policy_data.jsonl"

    with open(save_path, "w", encoding="utf-8") as f:
        for row in save_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(retrieval_path, "w", encoding="utf-8") as f:
        for row in retrieval_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "save_examples": len(save_data),
        "retrieval_examples": len(retrieval_data),
        "save_path": str(save_path),
        "retrieval_path": str(retrieval_path),
        "ready_for_training": (
            len(save_data) >= MEMORY_CONFIG["policy_min_save_examples"]
            and len(retrieval_data) >= MEMORY_CONFIG["policy_min_retrieval_examples"]
        ),
    }

    logger.info(
        "Policy data export: %d save examples, %d retrieval examples (ready=%s)",
        result["save_examples"], result["retrieval_examples"], result["ready_for_training"],
    )
    return result
