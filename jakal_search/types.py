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
    semantic_risk: float = 0.0
    embedding: np.ndarray | None = None
    cluster_id: int | None = None
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
            semantic_risk=self.semantic_risk,
            embedding=None if self.embedding is None else self.embedding.copy(),
            cluster_id=self.cluster_id,
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
            "semantic_risk": self.semantic_risk,
            "embedding_dim": self.embedding_dim,
            "cluster_id": None if self.cluster_id is None else int(self.cluster_id),
            "metadata": self.metadata,
            "source_profile": None if self.source_profile is None else self.source_profile.to_dict(),
        }
        if include_embedding and self.embedding is not None:
            payload["embedding"] = self.embedding.tolist()
        return payload


@dataclass(slots=True)
class SearchNode:
    node_id: str
    query: str
    depth: int
    parent_id: str | None = None
    score: float = 1.0
    cluster_label: str = ""
    status: str = "pending"
    stop_reason: str | None = None
    docs: list[SearchDocument] = field(default_factory=list)
    centroid: np.ndarray | None = None
    children: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchRequest:
    query: str
    max_depth: int | None = None
    max_total_nodes: int | None = None
    frontier_width: int | None = None
    results_per_query: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "max_depth": self.max_depth,
            "max_total_nodes": self.max_total_nodes,
            "frontier_width": self.frontier_width,
            "results_per_query": self.results_per_query,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SearchLogEvent:
    event_type: str
    node_id: str
    depth: int
    query: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "node_id": self.node_id,
            "depth": self.depth,
            "query": self.query,
            "payload": self.payload,
        }


@dataclass(slots=True)
class ExpansionContext:
    node_id: str
    query: str
    depth: int
    raw_document_ids: list[str] = field(default_factory=list)
    trusted_document_ids: list[str] = field(default_factory=list)
    unique_document_ids: list[str] = field(default_factory=list)
    cluster_ids: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    events: list[SearchLogEvent] = field(default_factory=list)

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> SearchLogEvent:
        event = SearchLogEvent(
            event_type=event_type,
            node_id=self.node_id,
            depth=self.depth,
            query=self.query,
            payload={} if payload is None else payload,
        )
        self.events.append(event)
        return event

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
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
class SearchTree:
    root_id: str
    request: SearchRequest | None = None
    nodes: dict[str, SearchNode] = field(default_factory=dict)
    expansions: dict[str, ExpansionContext] = field(default_factory=dict)
    logs: list[SearchLogEvent] = field(default_factory=list)

    def add_node(self, node: SearchNode) -> None:
        self.nodes[node.node_id] = node

    def add_expansion(self, context: ExpansionContext) -> None:
        self.expansions[context.node_id] = context

    def log_event(self, event: SearchLogEvent) -> None:
        self.logs.append(event)

    def ancestry_ids(self, node_id: str) -> list[str]:
        lineage: list[str] = []
        current = self.nodes[node_id]
        while current.parent_id is not None:
            lineage.append(current.parent_id)
            current = self.nodes[current.parent_id]
        return lineage

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "root_id": self.root_id,
            "request": None if self.request is None else self.request.to_dict(),
            "expansions": {node_id: context.to_dict() for node_id, context in self.expansions.items()},
            "logs": [event.to_dict() for event in self.logs],
            "nodes": {},
        }
        for node_id, node in self.nodes.items():
            payload["nodes"][node_id] = {
                "query": node.query,
                "depth": node.depth,
                "parent_id": node.parent_id,
                "score": node.score,
                "cluster_label": node.cluster_label,
                "status": node.status,
                "stop_reason": node.stop_reason,
                "children": node.children,
                "metrics": node.metrics,
                "docs": [doc.to_dict() for doc in node.docs],
            }
        return payload


@dataclass(slots=True)
class DocumentMemoryItem:
    node_id: str
    document_id: str
    url: str
    embedding: np.ndarray


@dataclass(slots=True)
class TopicMemoryItem:
    node_id: str
    embedding: np.ndarray


@dataclass(slots=True)
class TopicProposal:
    label: str
    query: str
    keywords: list[str]
    evidence_document_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClusterResult:
    cluster_id: int
    member_indexes: list[int]
    centroid: np.ndarray


@dataclass(slots=True)
class BranchMetrics:
    novelty: float
    trust: float
    scope: float
    support: float
    vector_consistency: float
    size_score: float
    drift: float


@dataclass(slots=True)
class BranchDecision:
    allowed: bool
    probability: float
    reason: str


@dataclass(slots=True)
class BranchState:
    topic: TopicProposal
    metrics: BranchMetrics
    decision: BranchDecision

    def to_node_metrics(self) -> dict[str, float | int | str]:
        return {
            "novelty": round(self.metrics.novelty, 4),
            "trust": round(self.metrics.trust, 4),
            "scope": round(self.metrics.scope, 4),
            "support": round(self.metrics.support, 4),
            "vector_consistency": round(self.metrics.vector_consistency, 4),
            "drift": round(self.metrics.drift, 4),
            "keyword_count": len(self.topic.keywords),
            "evidence_count": len(self.topic.evidence_document_ids),
            "decision_reason": self.decision.reason,
        }
