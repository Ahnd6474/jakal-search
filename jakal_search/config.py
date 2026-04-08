from dataclasses import dataclass, field
from pathlib import Path


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
class RetrievalConfig:
    enable_page_fetch: bool = True
    fetch_top_k: int = 6
    max_passages_per_doc: int = 3
    provider_pack: str = "auto"
    async_concurrency: int = 6
    per_host_delay_seconds: float = 0.4
    enable_cache: bool = True
    cache_ttl_hours: int = 72
    cache_dir: str = "outputs/cache/pages"
    lexical_weight: float = 0.35
    dense_weight: float = 0.4
    provider_weight: float = 0.15
    domain_weight: float = 0.1
    freshness_weight: float = 0.12
    entity_weight: float = 0.08
    passage_dedupe_threshold: float = 0.82


@dataclass(slots=True)
class TrustConfig:
    min_domain_trust: float = 0.22
    low_trust_semantic_threshold: float = 0.78
    domain_score_weight: float = 0.72
    semantic_score_weight: float = 0.28
    mlp_enabled_for_transformers: bool = True
    mlp_hidden_dim: int = 96
    mlp_epochs: int = 120
    mlp_learning_rate: float = 0.05
    mlp_weight_decay: float = 1e-4
    mlp_model_path: str | None = None
    mlp_threshold: float = 0.5
    citation_bonus_weight: float = 0.08
    author_bonus_weight: float = 0.05
    outbound_reference_bonus_weight: float = 0.05
    recency_bonus_weight: float = 0.04
    affiliate_penalty_weight: float = 0.09
    duplication_penalty_weight: float = 0.08
    blocked_domain_suffixes: tuple[str, ...] = ()
    domain_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_DOMAIN_TRUST.copy())
    trusted_prototypes: tuple[str, ...] = (
        "peer reviewed paper with citations, evidence, and transparent methodology",
        "official technical documentation with concrete examples and API references",
        "maintainer guide with reproducible steps, version notes, and tradeoff discussion",
        "government or academic publication grounded in verifiable sources",
    )
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
    source_diversity_weight: float = 0.8
    evidence_weight: float = 0.9
    falsehood_weight: float = 1.1
    freshness_weight: float = 0.5
    entity_weight: float = 0.55
    contradiction_weight: float = 0.75
    bias: float = -1.6
    mlp_model_path: str | None = None
    mlp_hidden_dim: int = 24
    mlp_threshold: float = 0.5


@dataclass(slots=True)
class TopicRerankerConfig:
    enabled: bool = True
    candidate_pool_size: int = 12
    model_path: str | None = None
    hidden_dim: int = 24


@dataclass(slots=True)
class ThemeTokenConfig:
    enabled: bool = True
    max_tokens: int = 6
    min_tokens: int = 2
    iterations: int = 4
    temperature: float = 0.35
    min_membership: float = 0.08
    graph_steps: int = 2
    edge_direction_weight: float = 0.35
    edge_epsilon: float = 1e-6


@dataclass(slots=True)
class FalsehoodConfig:
    enabled: bool = True
    model_path: str | None = None
    hidden_dim: int = 96
    threshold: float = 0.5
    penalty_weight: float = 0.55
    block_high_risk: bool = True


@dataclass(slots=True)
class EngineConfig:
    transformer_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    transformer_device: str = "auto"
    trust_model_path: str | None = None
    branch_model_path: str | None = None
    topic_reranker_model_path: str | None = None
    falsehood_model_path: str | None = None
    limits: SearchLimits = field(default_factory=SearchLimits)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    topic_reranker: TopicRerankerConfig = field(default_factory=TopicRerankerConfig)
    theme_tokens: ThemeTokenConfig = field(default_factory=ThemeTokenConfig)
    falsehood: FalsehoodConfig = field(default_factory=FalsehoodConfig)
