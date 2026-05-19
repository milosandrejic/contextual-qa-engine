def cohere_distance_to_relevance_percent(distance: float) -> int:
    """
    Convert a Cohere rerank distance value to a 0–100 relevance percentage.

    Cohere returns a relevance_score in [0.0, 1.0] (higher = more relevant).
    The reranker stores it as distance = 1.0 - relevance_score so the
    pipeline can use "lower is better" ordering consistently across all
    retrieval modes (cosine distance, BM25, RRF).

    To present a human-readable score to API consumers we undo the flip:

        relevance_score = 1.0 - distance      # undo the inversion
        percent        = relevance_score * 100 # scale to 0–100
        clamped        = max(0, min(100, ...)) # guard floating-point drift
    """
    relevance_score = 1.0 - distance
    percent = relevance_score * 100
    return max(0, min(100, round(percent)))
