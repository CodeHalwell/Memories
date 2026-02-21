"""Configuration for the Agent Memory System."""

from pathlib import Path

# Base data directory — all runtime data stored here
DATA_DIR = Path("data")
LOG_DIR = DATA_DIR / "logs" / "sessions"
DB_PATH = DATA_DIR / "memory.db"
GRAPH_DIR = DATA_DIR / "graph"
VECTOR_DIR = DATA_DIR / "vectors"
POLICY_DATA_DIR = DATA_DIR / "policy_data"

MEMORY_CONFIG = {
    # LLM
    "llm_model": "claude-sonnet-4-6",
    "llm_temperature": 0.2,

    # Save thresholds
    "save_confidence_threshold": 0.5,
    "fast_path_arousal": 0.85,
    "fast_path_surprise": 0.75,
    "max_keywords_per_memory": 10,

    # Save decision — retrieval gap awareness (A2.1)
    "gap_lookback_turns": 20,
    "gap_overlap_threshold": 0.3,
    "gap_threshold_reduction": 0.7,

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

    # Compaction — merge validation (A2.4)
    "merge_validation_queries": 5,
    "merge_degradation_tolerance": 0.15,

    # Compaction — generation gap guard (A2.3)
    "max_generation_gap_for_merge": 1,

    # Decay
    "decay_recency_weight": 0.6,
    "decay_frequency_weight": 0.4,
    "decay_halflife_days": 7,

    # Visual layer
    "visual_salience_threshold": 0.7,
    "clip_model": "ViT-B-32",

    # Embeddings
    "text_embedding_model": "all-MiniLM-L6-v2",

    # Dream exploration (A3)
    "dream_walk_count": 50,
    "dream_similarity_threshold": 0.7,
    "dream_max_new_edges": 20,
    "dream_cluster_min_size": 3,
    "dream_enabled": True,

    # Policy logging (A4)
    "policy_logging_enabled": True,
    "save_outcome_lookback_days": 30,
    "retrieval_outcome_followup_turns": 3,
    "retrieval_outcome_keyword_overlap": 0.5,

    # Policy training (A5, v2)
    "policy_min_save_examples": 1000,
    "policy_min_retrieval_examples": 500,
}
