from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import TopicRerankerConfig
from .embedding import TextEmbedder
from .types import SearchDocument
from .utils import cosine_similarity, mean_embedding


def topic_candidate_features(
    *,
    parent_query: str,
    candidate_text: str,
    docs: list[SearchDocument],
    embedder: TextEmbedder,
) -> np.ndarray:
    doc_vectors = [doc.embedding for doc in docs if doc.embedding is not None]
    centroid = mean_embedding(np.asarray(doc_vectors, dtype=np.float32))
    parent_vector, candidate_vector = embedder.embed([parent_query, candidate_text])
    similarities = [cosine_similarity(candidate_vector, vector) for vector in doc_vectors]
    parent_tokens = set(parent_query.lower().split())
    candidate_tokens = candidate_text.lower().split()
    overlap = len(parent_tokens & set(candidate_tokens)) / max(len(candidate_tokens), 1)
    mean_trust = float(np.mean([doc.trust_score for doc in docs])) if docs else 0.5
    return np.asarray(
        [
            cosine_similarity(parent_vector, candidate_vector),
            cosine_similarity(centroid, candidate_vector),
            float(np.mean(similarities)) if similarities else 0.0,
            float(np.max(similarities)) if similarities else 0.0,
            mean_trust,
            min(len(candidate_tokens), 6) / 6.0,
            overlap,
            1.0 if " ".join(candidate_tokens) in parent_query.lower() else 0.0,
        ],
        dtype=np.float32,
    )


class TopicQueryMLP(nn.Module):
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


class TopicQueryReranker:
    def __init__(self, config: TopicRerankerConfig, embedder: TextEmbedder) -> None:
        self._config = config
        self._embedder = embedder
        self._model: TopicQueryMLP | None = None
        self._maybe_load_model()

    def rank(
        self,
        *,
        parent_query: str,
        docs: list[SearchDocument],
        candidates: list[str],
    ) -> list[str]:
        if not candidates:
            return []
        scored = [(self._score(parent_query=parent_query, docs=docs, candidate=candidate), candidate) for candidate in candidates]
        scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return [candidate for _, candidate in scored]

    def _score(self, *, parent_query: str, docs: list[SearchDocument], candidate: str) -> float:
        features = topic_candidate_features(
            parent_query=parent_query,
            candidate_text=candidate,
            docs=docs,
            embedder=self._embedder,
        )
        if self._model is None:
            return float(
                (0.25 * features[1])
                + (0.35 * features[2])
                + (0.15 * features[3])
                + (0.15 * features[4])
                + (0.10 * (1.0 - features[6]))
            )
        with torch.no_grad():
            logits = self._model(torch.from_numpy(features).unsqueeze(0))
            return float(torch.sigmoid(logits).item())

    def _maybe_load_model(self) -> None:
        if not self._config.model_path:
            return
        path = Path(self._config.model_path)
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = TopicQueryMLP(int(payload["input_dim"]), int(payload["hidden_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._model = model
