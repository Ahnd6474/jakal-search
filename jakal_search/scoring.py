from __future__ import annotations

import numpy as np

from .config import ScoringConfig
from .types import BranchDecision, BranchMetrics, SearchTree
from .utils import cosine_similarity, logistic, mean_embedding


class VectorPathTracker:
    def __init__(self, config: ScoringConfig) -> None:
        self._config = config

    def score(self, tree: SearchTree, node_id: str, candidate_centroid: np.ndarray) -> tuple[float, float]:
        lineage = list(reversed(tree.ancestry_ids(node_id) + [node_id]))
        path_vectors = [
            tree.nodes[ancestor_id].centroid
            for ancestor_id in lineage
            if tree.nodes[ancestor_id].centroid is not None
        ]
        if not path_vectors:
            return 1.0, 0.0

        latest = path_vectors[-1]
        drift = 1.0 - cosine_similarity(candidate_centroid, latest)
        historical_shifts = [
            1.0 - cosine_similarity(path_vectors[index], path_vectors[index - 1])
            for index in range(1, len(path_vectors))
        ]
        shift_series = historical_shifts + [drift]
        volatility = float(np.std(shift_series)) if len(shift_series) > 1 else drift
        running_mean = mean_embedding(np.asarray(path_vectors, dtype=np.float32))
        alignment = max(0.0, cosine_similarity(candidate_centroid, running_mean))
        stability = max(0.0, 1.0 - (volatility / max(self._config.drift_soft_limit, 1e-6)))
        consistency = (0.55 * alignment) + (0.45 * stability)
        return consistency, drift


class BranchDecisionModel:
    def __init__(self, config: ScoringConfig) -> None:
        self._config = config

    def evaluate(self, metrics: BranchMetrics) -> BranchDecision:
        if metrics.novelty <= 0.0:
            return BranchDecision(allowed=False, probability=0.0, reason="no_new_information")
        if metrics.scope < 0.15:
            return BranchDecision(allowed=False, probability=0.0, reason="out_of_scope")
        if metrics.vector_consistency < self._config.min_vector_consistency and metrics.novelty < 0.35:
            return BranchDecision(allowed=False, probability=0.0, reason="unstable_vector_path")

        linear = (
            self._config.bias
            + (self._config.novelty_weight * metrics.novelty)
            + (self._config.trust_weight * metrics.trust)
            + (self._config.scope_weight * metrics.scope)
            + (self._config.support_weight * metrics.support)
            + (self._config.vector_weight * metrics.vector_consistency)
            + (self._config.size_weight * metrics.size_score)
        )
        probability = float(logistic(linear))
        if probability < self._config.min_continue_probability:
            return BranchDecision(allowed=False, probability=probability, reason="stop_classifier")
        return BranchDecision(allowed=True, probability=probability, reason="continue")
