from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SearchDocument:
    title: str
    snippet: str
    url: str
    source: str
    query: str
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    trust_score: float = 0.5
    semantic_risk: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.title, self.snippet) if part).strip()


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
class SearchTree:
    root_id: str
    nodes: dict[str, SearchNode] = field(default_factory=dict)

    def add_node(self, node: SearchNode) -> None:
        self.nodes[node.node_id] = node

    def ancestry_ids(self, node_id: str) -> list[str]:
        lineage: list[str] = []
        current = self.nodes[node_id]
        while current.parent_id is not None:
            lineage.append(current.parent_id)
            current = self.nodes[current.parent_id]
        return lineage

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"root_id": self.root_id, "nodes": {}}
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
                "docs": [
                    {
                        "title": doc.title,
                        "snippet": doc.snippet,
                        "url": doc.url,
                        "source": doc.source,
                        "query": doc.query,
                        "rank": doc.rank,
                        "trust_score": doc.trust_score,
                        "semantic_risk": doc.semantic_risk,
                    }
                    for doc in node.docs
                ],
            }
        return payload


@dataclass(slots=True)
class DocumentMemoryItem:
    node_id: str
    url: str
    embedding: np.ndarray


@dataclass(slots=True)
class TopicMemoryItem:
    node_id: str
    embedding: np.ndarray


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
