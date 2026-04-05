from __future__ import annotations

import numpy as np

from .config import TrustConfig
from .embedding import TextEmbedder
from .types import SearchDocument
from .utils import cosine_similarity


class SourceTrustScorer:
    def __init__(self, config: TrustConfig, embedder: TextEmbedder) -> None:
        self._config = config
        self._embedder = embedder
        self._prototype_embeddings: np.ndarray | None = None
        self._rejected_embeddings: list[np.ndarray] = []

    def score_domain(self, source: str) -> float:
        host = source.lower()
        for suffix, weight in self._config.domain_weights.items():
            normalized = suffix.lower()
            if normalized.startswith("."):
                if host.endswith(normalized):
                    return weight
                continue
            if host == normalized or host.endswith(f".{normalized}"):
                return weight
        return 0.55

    def assess_documents(
        self,
        documents: list[SearchDocument],
        embeddings: np.ndarray,
    ) -> tuple[list[SearchDocument], np.ndarray]:
        if not documents:
            return [], np.empty((0, 0), dtype=np.float32)

        if self._prototype_embeddings is None:
            self._prototype_embeddings = self._embedder.embed(list(self._config.suspicious_prototypes))

        kept_docs: list[SearchDocument] = []
        kept_vectors: list[np.ndarray] = []
        blocked_suffixes = tuple(item.lower() for item in self._config.blocked_domain_suffixes)

        for doc, vector in zip(documents, embeddings):
            host = doc.source.lower()
            if any(host == suffix or host.endswith(f".{suffix}") for suffix in blocked_suffixes):
                self._rejected_embeddings.append(vector)
                continue

            domain_score = self.score_domain(host)
            semantic_risk = 0.0
            if self._prototype_embeddings.size:
                semantic_risk = max(
                    semantic_risk,
                    max(cosine_similarity(vector, prototype) for prototype in self._prototype_embeddings),
                )
            if self._rejected_embeddings:
                semantic_risk = max(
                    semantic_risk,
                    max(cosine_similarity(vector, blocked) for blocked in self._rejected_embeddings),
                )

            doc.semantic_risk = semantic_risk
            doc.trust_score = max(0.0, min(1.0, (0.72 * domain_score) + (0.28 * (1.0 - semantic_risk))))

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
            kept_vectors.append(vector)

        if not kept_vectors:
            return [], np.empty((0, 0), dtype=np.float32)
        return kept_docs, np.asarray(kept_vectors, dtype=np.float32)
