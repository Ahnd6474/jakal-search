from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SourceProfile:
    host: str
    domain_score: float
    source_type: str
    matched_rule: str | None = None
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "domain_score": self.domain_score,
            "source_type": self.source_type,
            "matched_rule": self.matched_rule,
            "blocked": self.blocked,
        }


@dataclass(slots=True)
class PagePassage:
    text: str
    start: int = 0
    end: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


@dataclass(slots=True)
class ThemeToken:
    token_id: str
    label: str
    score: float
    query: str = ""
    asset_condition: str = ""
    base_score: float = 0.0
    state_norm: float = 0.0
    topic_norm: float = 0.0
    anchor_document_ids: list[str] = field(default_factory=list)
    anchor_titles: list[str] = field(default_factory=list)
    attention_document_ids: list[str] = field(default_factory=list)
    attention_titles: list[str] = field(default_factory=list)
    member_count: int = 0
    top_keywords: list[str] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    outgoing_edges: list[dict[str, Any]] = field(default_factory=list)
    state_vector: np.ndarray | None = None
    topic_vector: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "label": self.label,
            "query": self.query,
            "asset_condition": self.asset_condition,
            "score": self.score,
            "base_score": self.base_score,
            "state_norm": self.state_norm,
            "topic_norm": self.topic_norm,
            "anchor_document_ids": self.anchor_document_ids,
            "anchor_titles": self.anchor_titles,
            "attention_document_ids": self.attention_document_ids,
            "attention_titles": self.attention_titles,
            "member_count": self.member_count,
            "top_keywords": self.top_keywords,
            "candidate_scores": self.candidate_scores,
            "outgoing_edges": self.outgoing_edges,
        }


@dataclass(slots=True)
class SearchDocument:
    title: str
    snippet: str
    url: str
    source: str
    query: str
    rank: int
    document_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    trust_score: float = 0.5
    claim_falsehood_score: float = 0.0
    semantic_risk: float = 0.0
    retrieval_score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    provider_score: float = 0.0
    freshness_score: float = 0.0
    entity_score: float = 0.0
    published_at: str | None = None
    published_at_precision: str = "unknown"
    contradiction_label: str = "unknown"
    contradiction_cluster_id: int | None = None
    embedding: np.ndarray | None = None
    cluster_id: int | None = None
    passages: list[PagePassage] = field(default_factory=list)
    source_profile: SourceProfile | None = None

    def __post_init__(self) -> None:
        if not self.document_id:
            self.document_id = self.url or f"{self.query}:{self.rank}:{self.title}"
        if not self.content:
            self.content = self.snippet

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.title, self.content) if part).strip()

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.shape[0]) if self.embedding is not None else 0

    def clone(self) -> SearchDocument:
        return SearchDocument(
            document_id=self.document_id,
            title=self.title,
            snippet=self.snippet,
            url=self.url,
            source=self.source,
            query=self.query,
            rank=self.rank,
            content=self.content,
            metadata=self.metadata.copy(),
            trust_score=self.trust_score,
            claim_falsehood_score=self.claim_falsehood_score,
            semantic_risk=self.semantic_risk,
            retrieval_score=self.retrieval_score,
            dense_score=self.dense_score,
            lexical_score=self.lexical_score,
            provider_score=self.provider_score,
            freshness_score=self.freshness_score,
            entity_score=self.entity_score,
            published_at=self.published_at,
            published_at_precision=self.published_at_precision,
            contradiction_label=self.contradiction_label,
            contradiction_cluster_id=self.contradiction_cluster_id,
            embedding=None if self.embedding is None else self.embedding.copy(),
            cluster_id=self.cluster_id,
            passages=[PagePassage(**passage.to_dict()) for passage in self.passages],
            source_profile=None if self.source_profile is None else SourceProfile(**self.source_profile.to_dict()),
        )

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "query": self.query,
            "rank": self.rank,
            "trust_score": self.trust_score,
            "claim_falsehood_score": self.claim_falsehood_score,
            "semantic_risk": self.semantic_risk,
            "retrieval_score": self.retrieval_score,
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "provider_score": self.provider_score,
            "freshness_score": self.freshness_score,
            "entity_score": self.entity_score,
            "published_at": self.published_at,
            "published_at_precision": self.published_at_precision,
            "contradiction_label": self.contradiction_label,
            "contradiction_cluster_id": self.contradiction_cluster_id,
            "embedding_dim": self.embedding_dim,
            "cluster_id": None if self.cluster_id is None else int(self.cluster_id),
            "passages": [passage.to_dict() for passage in self.passages],
            "metadata": self.metadata,
            "source_profile": None if self.source_profile is None else self.source_profile.to_dict(),
        }
        if include_embedding and self.embedding is not None:
            payload["embedding"] = self.embedding.tolist()
        return payload


@dataclass(slots=True)
class SearchTopic:
    topic_id: str
    query: str
    depth: int
    parent_topic_id: str | None = None
    score: float = 1.0
    cluster_label: str = ""
    status: str = "pending"
    stop_reason: str | None = None
    docs: list[SearchDocument] = field(default_factory=list)
    centroid: np.ndarray | None = None
    child_topic_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    topic: TopicProposal | None = None
    theme_tokens: list[ThemeToken] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self.topic_id

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.topic_id = value

    @property
    def parent_id(self) -> str | None:
        return self.parent_topic_id

    @parent_id.setter
    def parent_id(self, value: str | None) -> None:
        self.parent_topic_id = value

    @property
    def children(self) -> list[str]:
        return self.child_topic_ids

    @children.setter
    def children(self, value: list[str]) -> None:
        self.child_topic_ids = value


@dataclass(slots=True)
class SearchRequest:
    query: str
    max_depth: int | None = None
    max_total_topics: int | None = None
    frontier_width: int | None = None
    results_per_query: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_total_nodes(self) -> int | None:
        return self.max_total_topics

    @max_total_nodes.setter
    def max_total_nodes(self, value: int | None) -> None:
        self.max_total_topics = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "max_depth": self.max_depth,
            "max_total_topics": self.max_total_topics,
            "frontier_width": self.frontier_width,
            "results_per_query": self.results_per_query,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchLogEvent:
    event_type: str
    topic_id: str
    depth: int
    query: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return self.topic_id

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.topic_id = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "topic_id": self.topic_id,
            "depth": self.depth,
            "query": self.query,
            "payload": self.payload,
        }


@dataclass(slots=True)
class ExpansionContext:
    topic_id: str
    query: str
    depth: int
    raw_document_ids: list[str] = field(default_factory=list)
    trusted_document_ids: list[str] = field(default_factory=list)
    unique_document_ids: list[str] = field(default_factory=list)
    cluster_ids: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    events: list[SearchLogEvent] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self.topic_id

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.topic_id = value

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> SearchLogEvent:
        event = SearchLogEvent(
            event_type=event_type,
            topic_id=self.topic_id,
            depth=self.depth,
            query=self.query,
            payload={} if payload is None else payload,
        )
        self.events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "query": self.query,
            "depth": self.depth,
            "raw_document_ids": self.raw_document_ids,
            "trusted_document_ids": self.trusted_document_ids,
            "unique_document_ids": self.unique_document_ids,
            "cluster_ids": [int(cluster_id) for cluster_id in self.cluster_ids],
            "stop_reason": self.stop_reason,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(slots=True)
class SearchRun:
    root_topic_id: str
    request: SearchRequest | None = None
    topics: dict[str, SearchTopic] = field(default_factory=dict)
    expansions: dict[str, ExpansionContext] = field(default_factory=dict)
    logs: list[SearchLogEvent] = field(default_factory=list)

    @property
    def root_id(self) -> str:
        return self.root_topic_id

    @root_id.setter
    def root_id(self, value: str) -> None:
        self.root_topic_id = value

    @property
    def nodes(self) -> dict[str, SearchTopic]:
        return self.topics

    def add_topic(self, topic: SearchTopic) -> None:
        self.topics[topic.topic_id] = topic

    def add_node(self, node: SearchTopic) -> None:
        self.add_topic(node)

    def add_expansion(self, context: ExpansionContext) -> None:
        self.expansions[context.topic_id] = context

    def log_event(self, event: SearchLogEvent) -> None:
        self.logs.append(event)

    def ancestry_topic_ids(self, topic_id: str) -> list[str]:
        lineage: list[str] = []
        current = self.topics[topic_id]
        while current.parent_topic_id is not None:
            lineage.append(current.parent_topic_id)
            current = self.topics[current.parent_topic_id]
        return lineage

    def ancestry_ids(self, node_id: str) -> list[str]:
        return self.ancestry_topic_ids(node_id)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "root_topic_id": self.root_topic_id,
            "request": None if self.request is None else self.request.to_dict(),
            "expansions": {topic_id: context.to_dict() for topic_id, context in self.expansions.items()},
            "logs": [event.to_dict() for event in self.logs],
            "topics": {},
        }
        for topic_id, topic in self.topics.items():
            payload["topics"][topic_id] = {
                "topic_id": topic.topic_id,
                "query": topic.query,
                "depth": topic.depth,
                "parent_topic_id": topic.parent_topic_id,
                "score": topic.score,
                "cluster_label": topic.cluster_label,
                "status": topic.status,
                "stop_reason": topic.stop_reason,
                "child_topic_ids": topic.child_topic_ids,
                "metrics": topic.metrics,
                "topic": None if topic.topic is None else topic.topic.to_dict(),
                "theme_tokens": [token.to_dict() for token in topic.theme_tokens],
                "docs": [doc.to_dict() for doc in topic.docs],
            }
        return payload


@dataclass(slots=True)
class DocumentMemoryItem:
    topic_id: str
    document_id: str
    url: str
    embedding: np.ndarray
    passage_signature: str = ""

    @property
    def node_id(self) -> str:
        return self.topic_id

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.topic_id = value


@dataclass(slots=True)
class TopicMemoryItem:
    topic_id: str
    embedding: np.ndarray

    @property
    def node_id(self) -> str:
        return self.topic_id

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.topic_id = value


@dataclass(slots=True)
class TopicProposal:
    label: str
    query: str
    keywords: list[str]
    evidence_document_ids: list[str] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "query": self.query,
            "keywords": self.keywords,
            "evidence_document_ids": self.evidence_document_ids,
            "candidate_scores": self.candidate_scores,
        }


@dataclass(slots=True)
class ClusterResult:
    cluster_id: int
    member_indexes: list[int]
    centroid: np.ndarray


@dataclass(slots=True)
class ExpansionMetrics:
    novelty: float
    trust: float
    scope: float
    support: float
    vector_consistency: float
    size_score: float
    drift: float
    source_diversity: float = 0.0
    evidence_coverage: float = 0.0
    falsehood_penalty: float = 0.0
    freshness: float = 0.0
    entity_alignment: float = 0.0
    contradiction_penalty: float = 0.0


@dataclass(slots=True)
class ExpansionDecision:
    allowed: bool
    probability: float
    reason: str


@dataclass(slots=True)
class ExpansionState:
    topic: TopicProposal
    metrics: ExpansionMetrics
    decision: ExpansionDecision

    def to_topic_metrics(self) -> dict[str, float | int | str]:
        return {
            "novelty": round(self.metrics.novelty, 4),
            "trust": round(self.metrics.trust, 4),
            "scope": round(self.metrics.scope, 4),
            "support": round(self.metrics.support, 4),
            "vector_consistency": round(self.metrics.vector_consistency, 4),
            "drift": round(self.metrics.drift, 4),
            "source_diversity": round(self.metrics.source_diversity, 4),
            "evidence_coverage": round(self.metrics.evidence_coverage, 4),
            "falsehood_penalty": round(self.metrics.falsehood_penalty, 4),
            "freshness": round(self.metrics.freshness, 4),
            "entity_alignment": round(self.metrics.entity_alignment, 4),
            "contradiction_penalty": round(self.metrics.contradiction_penalty, 4),
            "keyword_count": len(self.topic.keywords),
            "evidence_count": len(self.topic.evidence_document_ids),
            "decision_reason": self.decision.reason,
        }

    def to_node_metrics(self) -> dict[str, float | int | str]:
        return self.to_topic_metrics()


SearchTree = SearchRun
SearchNode = SearchTopic

BranchMetrics = ExpansionMetrics
BranchDecision = ExpansionDecision
BranchState = ExpansionState
