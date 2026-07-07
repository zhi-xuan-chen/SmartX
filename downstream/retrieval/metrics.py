"""Image-text retrieval metrics adapted from the evaluation repository."""

from __future__ import annotations

import numpy as np


def ranks_from_similarity(similarity: np.ndarray, direction: str) -> np.ndarray:
    """Return 1-indexed ranks for diagonal matches in a square similarity matrix."""
    similarity = np.asarray(similarity)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(
            "similarity must be a square [num_cases, num_cases] matrix, "
            f"got {similarity.shape}"
        )
    if direction == "image_to_text":
        scores = similarity
    elif direction == "text_to_image":
        scores = similarity.T
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    target_scores = np.diag(scores)
    return (scores > target_scores[:, None]).sum(axis=1).astype(np.int64) + 1


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        raise ValueError("Cannot compute recall for an empty rank array.")
    return float(np.mean(ranks <= int(k)))


def metrics_from_ranks(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 50, 100)) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size == 0:
        raise ValueError("Cannot compute metrics for an empty rank array.")
    payload = {f"recall@{k}": recall_at_k(ranks, k) for k in ks}
    payload.update(
        {
            "precision@1": payload.get("recall@1", recall_at_k(ranks, 1)),
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "num_queries": int(ranks.size),
        }
    )
    return payload


def metrics_from_similarity(
    similarity: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 50, 100),
) -> dict[str, dict[str, float]]:
    return {
        direction: metrics_from_ranks(ranks_from_similarity(similarity, direction), ks=ks)
        for direction in ("image_to_text", "text_to_image")
    }
