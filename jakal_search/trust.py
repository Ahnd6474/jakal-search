from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn

from .config import TrustConfig
from .embedding import TextEmbedder
from .types import SearchDocument, SourceProfile
from .utils import cosine_similarity

SOURCE_TYPE_ORDER = (
    "government",
    "academic",
    "research",
    "technical",
    "news",
    "blog",
    "web",
)


class EmbeddingTrustMLP(nn.Module):
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


def normalize_source_host(source: str) -> str:
    value = source.strip().lower()
    if not value:
        return ""
    if "://" not in value:
        return value.split("/", 1)[0]
    parsed = urlparse(value)
    return parsed.netloc.lower() or value


def infer_source_type(host: str) -> str:
    if host.endswith(".gov"):
        return "government"
    if host.endswith(".edu"):
        return "academic"
    if any(domain in host for domain in ("arxiv.org", "acm.org", "ieee.org", "nature.com", "science.org")):
        return "research"
    if any(domain in host for domain in ("github.com", "docs.python.org", "openai.com", "readthedocs.io")):
        return "technical"
    if any(domain in host for domain in ("bbc.com", "bbc.co.uk", "npr.org", "reuters.com", "apnews.com")):
        return "news"
    if any(domain in host for domain in ("medium.com", "substack.com", "blogspot.com", "wordpress.com")):
        return "blog"
    return "web"


def is_blocked_host(host: str, config: TrustConfig) -> bool:
    blocked_suffixes = tuple(item.lower() for item in config.blocked_domain_suffixes)
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in blocked_suffixes)


def resolve_source_profile(config: TrustConfig, source: str) -> SourceProfile:
    host = normalize_source_host(source)
    blocked = is_blocked_host(host, config)
    for suffix, weight in config.domain_weights.items():
        normalized = suffix.lower()
        if normalized.startswith("."):
            if host.endswith(normalized):
                return SourceProfile(
                    host=host,
                    domain_score=weight,
                    source_type=infer_source_type(host),
                    matched_rule=normalized,
                    blocked=blocked,
                )
            continue
        if host == normalized or host.endswith(f".{normalized}"):
            return SourceProfile(
                host=host,
                domain_score=weight,
                source_type=infer_source_type(host),
                matched_rule=normalized,
                blocked=blocked,
            )
    return SourceProfile(
        host=host,
        domain_score=0.55,
        source_type=infer_source_type(host),
        matched_rule=None,
        blocked=blocked,
    )


def build_source_feature_vector(profile: SourceProfile) -> np.ndarray:
    vector = np.zeros(14, dtype=np.float32)
    vector[0] = float(profile.domain_score)
    vector[1] = 1.0 if profile.blocked else 0.0
    if profile.matched_rule is not None:
        vector[2] = 1.0
    for index, source_type in enumerate(SOURCE_TYPE_ORDER, start=3):
        if profile.source_type == source_type:
            vector[index] = 1.0
            break
    host = profile.host
    vector[10] = 1.0 if host.endswith(".org") else 0.0
    vector[11] = 1.0 if host.endswith(".com") else 0.0
    vector[12] = 1.0 if host.endswith(".gov") or host.endswith(".edu") else 0.0
    vector[13] = min(host.count("."), 4) / 4.0
    return vector


def build_trust_feature_vector(embedding: np.ndarray, profile: SourceProfile) -> np.ndarray:
    return np.concatenate([embedding.astype(np.float32), build_source_feature_vector(profile)], axis=0)


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
        return resolve_source_profile(self._config, source)

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
            profile = self.resolve_source_profile(doc.source or doc.url)
            doc.source_profile = profile
            if profile.blocked:
                self._rejected_embeddings.append(vector)
                continue

            if self._use_embedding_mlp and self._embedding_trust_head is not None:
                trust_score = self._score_with_embedding_mlp(vector, profile)
                semantic_risk = max(0.0, min(1.0, 1.0 - trust_score))
            else:
                semantic_risk, semantic_score = self._semantic_scores(vector)
                trust_score = max(
                    0.0,
                    min(
                        1.0,
                        (self._config.domain_score_weight * profile.domain_score)
                        + (self._config.semantic_score_weight * semantic_score),
                    ),
                )

            if self._rejected_embeddings and not self._use_embedding_mlp:
                semantic_risk = max(
                    semantic_risk,
                    max(cosine_similarity(vector, blocked) for blocked in self._rejected_embeddings),
                )

            doc.semantic_risk = semantic_risk
            doc.trust_score = trust_score

            if self._use_embedding_mlp and self._embedding_trust_head is not None:
                should_reject = trust_score < self._embedding_threshold
            else:
                should_reject = (
                    (
                        profile.domain_score < self._config.min_domain_trust
                        and semantic_risk >= self._config.low_trust_semantic_threshold
                    )
                    or (
                        trust_score < 0.38
                        and semantic_risk >= (self._config.low_trust_semantic_threshold - 0.12)
                    )
                )

            if should_reject:
                self._rejected_embeddings.append(vector)
                continue
            kept_docs.append(doc)
        return kept_docs

    def _semantic_scores(self, vector: np.ndarray) -> tuple[float, float]:
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

        trusted_hosts = ("docs.python.org", "acm.org", "ieee.org", "npr.org")
        suspicious_hosts = ("blogspot.com", "wordpress.com", "medium.com", "substack.com")
        trusted_rows = [
            build_trust_feature_vector(
                embedding,
                resolve_source_profile(self._config, trusted_hosts[index % len(trusted_hosts)]),
            )
            for index, embedding in enumerate(trusted_embeddings)
        ]
        suspicious_rows = [
            build_trust_feature_vector(
                embedding,
                resolve_source_profile(self._config, suspicious_hosts[index % len(suspicious_hosts)]),
            )
            for index, embedding in enumerate(suspicious_embeddings)
        ]
        train_x = np.asarray([*trusted_rows, *suspicious_rows], dtype=np.float32)
        train_y = np.concatenate(
            [
                np.ones(len(trusted_rows), dtype=np.float32),
                np.zeros(len(suspicious_rows), dtype=np.float32),
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

    def _score_with_embedding_mlp(self, vector: np.ndarray, profile: SourceProfile) -> float:
        if self._embedding_trust_head is None:
            return 0.5
        feature_row = build_trust_feature_vector(vector, profile)
        with torch.no_grad():
            logits = self._embedding_trust_head(torch.from_numpy(feature_row).unsqueeze(0))
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
