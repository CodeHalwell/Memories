"""Configuration for the Agent Memory System."""

from pathlib import Path

# Base data directory — all runtime data stored here
DATA_DIR = Path("data")
LOG_DIR = DATA_DIR / "logs" / "sessions"
DB_PATH = DATA_DIR / "memory.db"
GRAPH_DIR = DATA_DIR / "graph"
VECTOR_DIR = DATA_DIR / "vectors"

MEMORY_CONFIG = {
    # LLM
    "llm_model": "claude-sonnet-4-6",
    "llm_temperature": 0.2,

    # Save thresholds
    "save_confidence_threshold": 0.5,
    "fast_path_arousal": 0.85,
    "fast_path_surprise": 0.75,
    "max_keywords_per_memory": 10,

    # Retrieval
    "retrieval_layers": ["grep", "keyword", "semantic"],
    "graph_traversal_depth": 2,
    "mood_congruent_weight": 0.2,
    "top_k_per_layer": 5,

    # Compaction
    "hot_tier_threshold": 500,
    "compaction_candidate_threshold": 0.7,
    "keyword_overlap_merge_threshold": 0.6,
    "valence_merge_exclusion_delta": 0.6,

    # Decay
    "decay_recency_weight": 0.6,
    "decay_frequency_weight": 0.4,
    "decay_halflife_days": 7,

    # Visual layer
    "visual_salience_threshold": 0.7,
    "clip_model": "ViT-B-32",

    # Embeddings
    "text_embedding_model": "all-MiniLM-L6-v2",
}
