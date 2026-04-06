from __future__ import annotations

import numpy as np

from .config import TrustConfig
from .embedding import TextEmbedder
from .types import SearchDocument, SourceProfile
from .utils import cosine_similarity


class SourceTrustScorer:
    def __init__(self, config: TrustConfig, embedder: TextEmbedder) -> None:
        self._config = config
        self._embedder = embedder
        self._prototype_embeddings: np.ndarray | None = None
        self._rejected_embeddings: list[np.ndarray] = []

    def reset(self) -> None:
        self._prototype_embeddings = None
        self._rejected_embeddings = []

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
        return kept_docs
