from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ScoringConfig
from .types import ExpansionDecision, ExpansionMetrics, SearchRun
from .utils import cosine_similarity, logistic, mean_embedding


def expansion_metrics_to_vector(metrics: ExpansionMetrics) -> np.ndarray:
    return np.asarray(
        [
            metrics.novelty,
            metrics.trust,
            metrics.scope,
            metrics.support,
            metrics.vector_consistency,
            metrics.size_score,
            metrics.drift,
            metrics.source_diversity,
            metrics.evidence_coverage,
            metrics.falsehood_penalty,
            metrics.freshness,
            metrics.entity_alignment,
            metrics.contradiction_penalty,
        ],
        dtype=np.float32,
    )


class ExpansionDecisionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 4), 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


class VectorPathTracker:
    def __init__(self, config: ScoringConfig) -> None:
        self._config = config

    def score(self, run: SearchRun, topic_id: str, candidate_centroid: np.ndarray) -> tuple[float, float]:
        lineage = list(reversed(run.ancestry_topic_ids(topic_id) + [topic_id]))
        path_vectors = [
            run.topics[ancestor_id].centroid
            for ancestor_id in lineage
            if run.topics[ancestor_id].centroid is not None
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


class ExpansionDecisionModel:
    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._head: ExpansionDecisionMLP | None = None
        self._threshold = config.mlp_threshold
        self._maybe_load_model()

    def evaluate(self, metrics: ExpansionMetrics) -> ExpansionDecision:
        if metrics.novelty <= 0.0:
            return ExpansionDecision(allowed=False, probability=0.0, reason="no_new_information")
        if metrics.scope < 0.15:
            return ExpansionDecision(allowed=False, probability=0.0, reason="out_of_scope")
        if metrics.vector_consistency < self._config.min_vector_consistency and metrics.novelty < 0.35:
            return ExpansionDecision(allowed=False, probability=0.0, reason="unstable_vector_path")
        if metrics.falsehood_penalty >= 0.72 and metrics.trust < 0.55:
            return ExpansionDecision(allowed=False, probability=0.0, reason="high_falsehood_risk")
        if metrics.contradiction_penalty >= 0.78 and metrics.evidence_coverage < 0.2:
            return ExpansionDecision(allowed=False, probability=0.0, reason="weakly_supported_contradiction")

        if self._head is not None:
            probability = self._predict_probability(metrics)
            if probability < self._threshold:
                return ExpansionDecision(allowed=False, probability=probability, reason="expansion_model")
            return ExpansionDecision(allowed=True, probability=probability, reason="expansion_model")

        linear = (
            self._config.bias
            + (self._config.novelty_weight * metrics.novelty)
            + (self._config.trust_weight * metrics.trust)
            + (self._config.scope_weight * metrics.scope)
            + (self._config.support_weight * metrics.support)
            + (self._config.vector_weight * metrics.vector_consistency)
            + (self._config.size_weight * metrics.size_score)
            + (self._config.source_diversity_weight * metrics.source_diversity)
            + (self._config.evidence_weight * min(metrics.evidence_coverage, 1.0))
            + (self._config.freshness_weight * metrics.freshness)
            + (self._config.entity_weight * metrics.entity_alignment)
            - (self._config.falsehood_weight * metrics.falsehood_penalty)
            - (self._config.contradiction_weight * metrics.contradiction_penalty)
        )
        probability = float(logistic(linear))
        if probability < self._config.min_continue_probability:
            return ExpansionDecision(allowed=False, probability=probability, reason="stop_classifier")
        return ExpansionDecision(allowed=True, probability=probability, reason="continue")

    def _predict_probability(self, metrics: ExpansionMetrics) -> float:
        if self._head is None:
            return 0.5
        vector = self._fit_feature_vector(expansion_metrics_to_vector(metrics), self._head.layers[0].in_features)
        with torch.no_grad():
            logits = self._head(torch.from_numpy(vector).unsqueeze(0))
            return float(torch.sigmoid(logits).item())

    def _maybe_load_model(self) -> None:
        model_path = self._config.mlp_model_path
        if not model_path:
            return
        path = Path(model_path)
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = ExpansionDecisionMLP(int(payload["input_dim"]), int(payload["hidden_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._head = model
        self._threshold = float(payload.get("threshold", self._config.mlp_threshold))

    def _fit_feature_vector(self, vector: np.ndarray, input_dim: int) -> np.ndarray:
        if vector.shape[0] == input_dim:
            return vector
        if vector.shape[0] > input_dim:
            return vector[:input_dim]
        return np.pad(vector, (0, input_dim - vector.shape[0]), mode="constant")


branch_metrics_to_vector = expansion_metrics_to_vector
BranchDecisionMLP = ExpansionDecisionMLP
BranchDecisionModel = ExpansionDecisionModel
