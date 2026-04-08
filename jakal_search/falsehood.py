from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import FalsehoodConfig
from .types import SearchDocument


class ClaimFalsehoodMLP(nn.Module):
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


class ClaimFalsehoodScorer:
    def __init__(self, config: FalsehoodConfig) -> None:
        self._config = config
        self._model: ClaimFalsehoodMLP | None = None
        self._threshold = config.threshold
        self._maybe_load_model()

    @property
    def enabled(self) -> bool:
        return self._model is not None and self._config.enabled

    def assess_documents(self, documents: list[SearchDocument]) -> list[SearchDocument]:
        if not self.enabled or not documents:
            return documents

        vectors = [doc.embedding for doc in documents if doc.embedding is not None and doc.embedding.size > 0]
        if len(vectors) != len(documents):
            return documents
        scores = self.score_vectors(np.asarray(vectors, dtype=np.float32))
        kept_documents: list[SearchDocument] = []
        for doc, score in zip(documents, scores):
            doc.claim_falsehood_score = float(score)
            doc.trust_score = max(
                0.0,
                min(1.0, doc.trust_score * (1.0 - (self._config.penalty_weight * doc.claim_falsehood_score))),
            )
            if self._config.block_high_risk and doc.claim_falsehood_score >= self._threshold:
                continue
            kept_documents.append(doc)
        return kept_documents

    def score_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if self._model is None or vectors.size == 0:
            return np.zeros((len(vectors),), dtype=np.float32)
        with torch.no_grad():
            logits = self._model(torch.from_numpy(vectors.astype(np.float32)))
            return torch.sigmoid(logits).numpy().astype(np.float32)

    def _maybe_load_model(self) -> None:
        if not self._config.model_path:
            return
        path = Path(self._config.model_path)
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = ClaimFalsehoodMLP(int(payload["input_dim"]), int(payload["hidden_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._model = model
        self._threshold = float(payload.get("threshold", self._config.threshold))
