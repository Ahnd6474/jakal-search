from __future__ import annotations

import numpy as np
import torch
from torch import nn
from pathlib import Path

from .config import TrustConfig
from .embedding import TextEmbedder
from .types import SearchDocument, SourceProfile
from .utils import cosine_similarity


class EmbeddingTrustMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


class SourceTrustScorer:
    def __init__(self, config: TrustConfig, embedder: TextEmbedder) -> None:
        self._config = config
        self._embedder = embedder
        self._use_embedding_mlp = bool(getattr(embedder, "using_transformer", False)) and config.mlp_enabled_for_transformers
        self._loaded_model = False
        self._prototype_embeddings: np.ndarray | None = None
        self._trusted_prototype_embeddings: np.ndarray | None = None
        self._rejected_embeddings: list[np.ndarray] = []
        self._embedding_trust_head: EmbeddingTrustMLP | None = None
        self._embedding_threshold = config.mlp_threshold
        self._maybe_load_trained_head()

    def reset(self) -> None:
        self._prototype_embeddings = None
        self._trusted_prototype_embeddings = None
        self._rejected_embeddings = []
        if not self._loaded_model:
            self._embedding_trust_head = None

    def resolve_source_profile(self, source: str) -> SourceProfile:
        host = source.lower()
        blocked = self._is_blocked(host)
        for suffix, weight in self._config.domain_weights.items():
            normalized = suffix.lower()
            if normalized.startswith("."):
                if host.endswith(normalized):
                    return SourceProfile(
                        host=host,
                        domain_score=weight,
                        source_type=self._infer_source_type(host),
                        matched_rule=normalized,
                        blocked=blocked,
                    )
                continue
            if host == normalized or host.endswith(f".{normalized}"):
                return SourceProfile(
                    host=host,
                    domain_score=weight,
                    source_type=self._infer_source_type(host),
                    matched_rule=normalized,
                    blocked=blocked,
                )
        return SourceProfile(
            host=host,
            domain_score=0.55,
            source_type=self._infer_source_type(host),
            matched_rule=None,
            blocked=blocked,
        )

    def _is_blocked(self, host: str) -> bool:
        blocked_suffixes = tuple(item.lower() for item in self._config.blocked_domain_suffixes)
        return any(host == suffix or host.endswith(f".{suffix}") for suffix in blocked_suffixes)

    def _infer_source_type(self, host: str) -> str:
        if host.endswith(".gov"):
            return "government"
        if host.endswith(".edu"):
            return "academic"
        if any(domain in host for domain in ("arxiv.org", "acm.org", "ieee.org", "nature.com", "science.org")):
            return "research"
        if any(domain in host for domain in ("github.com", "docs.python.org", "openai.com")):
            return "technical"
        if any(domain in host for domain in ("medium.com", "substack.com", "blogspot.com", "wordpress.com")):
            return "blog"
        return "web"

    def assess_documents(
        self,
        documents: list[SearchDocument],
    ) -> list[SearchDocument]:
        if not documents:
            return []

        if self._prototype_embeddings is None:
            self._prototype_embeddings = self._embedder.embed(list(self._config.suspicious_prototypes))
        if self._use_embedding_mlp and not self._loaded_model and self._trusted_prototype_embeddings is None:
            self._trusted_prototype_embeddings = self._embedder.embed(list(self._config.trusted_prototypes))
            self._embedding_trust_head = self._fit_embedding_trust_head(
                self._trusted_prototype_embeddings,
                self._prototype_embeddings,
            )

        kept_docs: list[SearchDocument] = []

        for doc in documents:
            vector = doc.embedding
            if vector is None or vector.size == 0:
                continue
            profile = self.resolve_source_profile(doc.source)
            doc.source_profile = profile
            if profile.blocked:
                self._rejected_embeddings.append(vector)
                continue

            domain_score = profile.domain_score
            semantic_risk, semantic_score = self._semantic_scores(vector)
            if self._rejected_embeddings:
                semantic_risk = max(
                    semantic_risk,
                    max(cosine_similarity(vector, blocked) for blocked in self._rejected_embeddings),
                )

            doc.semantic_risk = semantic_risk
            doc.trust_score = max(
                0.0,
                min(
                    1.0,
                    (self._config.domain_score_weight * domain_score)
                    + (self._config.semantic_score_weight * semantic_score),
                ),
            )

            should_reject = (
                (
                    domain_score < self._config.min_domain_trust
                    and semantic_risk >= self._config.low_trust_semantic_threshold
                )
                or (
                    doc.trust_score < 0.38
                    and semantic_risk >= (self._config.low_trust_semantic_threshold - 0.12)
                )
            )
            if should_reject:
                self._rejected_embeddings.append(vector)
                continue

            kept_docs.append(doc)
        return kept_docs

    def _semantic_scores(self, vector: np.ndarray) -> tuple[float, float]:
        if self._use_embedding_mlp and self._embedding_trust_head is not None:
            semantic_score = self._score_with_embedding_mlp(vector)
            return max(0.0, min(1.0, 1.0 - semantic_score)), semantic_score

        semantic_risk = 0.0
        if self._prototype_embeddings is not None and self._prototype_embeddings.size:
            semantic_risk = max(
                semantic_risk,
                max(cosine_similarity(vector, prototype) for prototype in self._prototype_embeddings),
            )
        return semantic_risk, max(0.0, min(1.0, 1.0 - semantic_risk))

    def _fit_embedding_trust_head(
        self,
        trusted_embeddings: np.ndarray,
        suspicious_embeddings: np.ndarray,
    ) -> EmbeddingTrustMLP | None:
        if trusted_embeddings.size == 0 or suspicious_embeddings.size == 0:
            return None

        train_x = np.concatenate([trusted_embeddings, suspicious_embeddings], axis=0).astype(np.float32)
        train_y = np.concatenate(
            [
                np.ones(len(trusted_embeddings), dtype=np.float32),
                np.zeros(len(suspicious_embeddings), dtype=np.float32),
            ]
        )

        torch.manual_seed(0)
        model = EmbeddingTrustMLP(train_x.shape[1], self._config.mlp_hidden_dim)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._config.mlp_learning_rate,
            weight_decay=self._config.mlp_weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        inputs = torch.from_numpy(train_x)
        targets = torch.from_numpy(train_y)

        model.train()
        for _ in range(self._config.mlp_epochs):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def _score_with_embedding_mlp(self, vector: np.ndarray) -> float:
        if self._embedding_trust_head is None:
            return 0.5
        with torch.no_grad():
            logits = self._embedding_trust_head(torch.from_numpy(vector.astype(np.float32)).unsqueeze(0))
            return float(torch.sigmoid(logits).item())

    def _maybe_load_trained_head(self) -> None:
        model_path = self._config.mlp_model_path
        if not self._use_embedding_mlp or not model_path:
            return
        path = Path(model_path)
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = EmbeddingTrustMLP(int(payload["input_dim"]), int(payload["hidden_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._embedding_trust_head = model
        self._embedding_threshold = float(payload.get("threshold", self._config.mlp_threshold))
        self._loaded_model = True
