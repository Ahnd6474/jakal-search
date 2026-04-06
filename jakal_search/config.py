from dataclasses import dataclass, field


DEFAULT_DOMAIN_TRUST = {
    "wikipedia.org": 0.92,
    "arxiv.org": 0.90,
    "nature.com": 0.95,
    "science.org": 0.95,
    "acm.org": 0.92,
    "ieee.org": 0.92,
    "github.com": 0.82,
    "docs.python.org": 0.95,
    "openai.com": 0.92,
    ".gov": 0.96,
    ".edu": 0.88,
    "substack.com": 0.48,
    "medium.com": 0.45,
    "blogspot.com": 0.30,
    "wordpress.com": 0.35,
}


@dataclass(slots=True)
class SearchLimits:
    max_depth: int = 3
    max_total_nodes: int = 24
    frontier_width: int = 6
    max_children_per_node: int = 4
    results_per_query: int = 12
    min_results: int = 4
    min_cluster_size: int = 3


@dataclass(slots=True)
class SimilarityConfig:
    dedupe_threshold: float = 0.92
    novelty_threshold: float = 0.18
    scope_threshold: float = 0.35


@dataclass(slots=True)
class TrustConfig:
    min_domain_trust: float = 0.22
    low_trust_semantic_threshold: float = 0.78
    blocked_domain_suffixes: tuple[str, ...] = ()
    domain_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_DOMAIN_TRUST.copy())
    suspicious_prototypes: tuple[str, ...] = (
        "secret miracle cure shocking truth guaranteed results click now",
        "conspiracy cover up they do not want you to know this",
        "free crypto giveaway urgent act now limited offer",
        "spam content farm copied article with no evidence",
    )


@dataclass(slots=True)
class ScoringConfig:
    min_continue_probability: float = 0.48
    min_vector_consistency: float = 0.32
    drift_soft_limit: float = 0.45
    novelty_weight: float = 1.8
    trust_weight: float = 1.1
    scope_weight: float = 1.0
    support_weight: float = 0.8
    vector_weight: float = 1.2
    size_weight: float = 0.7
    bias: float = -1.6


@dataclass(slots=True)
class EngineConfig:
    transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    transformer_device: str = "auto"
    limits: SearchLimits = field(default_factory=SearchLimits)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
