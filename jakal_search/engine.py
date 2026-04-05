from __future__ import annotations

import json
from itertools import count

import numpy as np

from .clustering import DensityClusterer
from .config import EngineConfig
from .embedding import SentenceTransformerEmbedder, TextEmbedder
from .providers import DuckDuckGoHtmlProvider, SearchProvider
from .scoring import BranchDecisionModel, VectorPathTracker
from .topics import KeywordTopicBuilder
from .trust import SourceTrustScorer
from .types import (
    BranchMetrics,
    DocumentMemoryItem,
    SearchDocument,
    SearchNode,
    SearchTree,
    TopicMemoryItem,
)
from .utils import cosine_similarity, mean_embedding


class SimilarityDeduper:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def dedupe(
        self,
        docs: list[SearchDocument],
        embeddings: np.ndarray,
        ancestry_ids: set[str],
        memory: list[DocumentMemoryItem],
    ) -> tuple[list[SearchDocument], np.ndarray]:
        kept_docs: list[SearchDocument] = []
        kept_vectors: list[np.ndarray] = []

        for doc, vector in zip(docs, embeddings):
            if self._is_local_duplicate(vector, kept_vectors):
                continue
            if self._is_global_duplicate(vector, ancestry_ids, memory):
                continue
            kept_docs.append(doc)
            kept_vectors.append(vector)

        if not kept_vectors:
            return [], np.empty((0, 0), dtype=np.float32)
        return kept_docs, np.asarray(kept_vectors, dtype=np.float32)

    def _is_local_duplicate(self, vector: np.ndarray, kept_vectors: list[np.ndarray]) -> bool:
        return any(cosine_similarity(vector, other) >= self._threshold for other in kept_vectors)

    def _is_global_duplicate(
        self,
        vector: np.ndarray,
        ancestry_ids: set[str],
        memory: list[DocumentMemoryItem],
    ) -> bool:
        for item in memory:
            if item.node_id in ancestry_ids:
                continue
            if cosine_similarity(vector, item.embedding) >= self._threshold:
                return True
        return False


class SearchTreeEngine:
    def __init__(
        self,
        provider: SearchProvider,
        embedder: TextEmbedder,
        config: EngineConfig | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.provider = provider
        self.embedder = embedder
        self.clusterer = DensityClusterer(self.config.limits.min_cluster_size)
        self.trust_scorer = SourceTrustScorer(self.config.trust, embedder)
        self.topic_builder = KeywordTopicBuilder()
        self.path_tracker = VectorPathTracker(self.config.scoring)
        self.decision_model = BranchDecisionModel(self.config.scoring)
        self.deduper = SimilarityDeduper(self.config.similarity.dedupe_threshold)
        self._id_counter = count()
        self._document_memory: list[DocumentMemoryItem] = []
        self._topic_memory: list[TopicMemoryItem] = []
        self._root_query_vector: np.ndarray | None = None

    def run(self, root_query: str) -> SearchTree:
        root_embedding = self.embedder.embed([root_query])
        self._root_query_vector = root_embedding[0]

        root = SearchNode(
            node_id=self._next_node_id(),
            query=root_query,
            depth=0,
            centroid=self._root_query_vector,
            status="pending",
            score=1.0,
        )
        tree = SearchTree(root_id=root.node_id)
        tree.add_node(root)
        frontier = [root.node_id]

        while frontier and len(tree.nodes) < self.config.limits.max_total_nodes:
            frontier.sort(key=lambda node_id: tree.nodes[node_id].score, reverse=True)
            current_id = frontier.pop(0)
            current = tree.nodes[current_id]

            new_children = self._expand_node(tree, current)
            frontier.extend(new_children)
            frontier = self._prune_frontier(tree, frontier)

        for pending_id in frontier:
            pending = tree.nodes[pending_id]
            if pending.status == "pending":
                pending.status = "pruned"
                pending.stop_reason = "global_node_limit"

        return tree

    def dumps(self, tree: SearchTree) -> str:
        return json.dumps(tree.to_dict(), ensure_ascii=False, indent=2)

    def _expand_node(self, tree: SearchTree, node: SearchNode) -> list[str]:
        if node.depth >= self.config.limits.max_depth:
            node.status = "stopped"
            node.stop_reason = "max_depth"
            return []

        results = self.provider.search(node.query, self.config.limits.results_per_query)
        if len(results) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "too_few_results"
            return []

        embeddings = self.embedder.embed([doc.text for doc in results])
        trusted_docs, trusted_vectors = self.trust_scorer.assess_documents(results, embeddings)
        if len(trusted_docs) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "trust_filter_exhausted_results"
            return []

        ancestry_ids = set(tree.ancestry_ids(node.node_id) + [node.node_id])
        unique_docs, unique_vectors = self.deduper.dedupe(
            trusted_docs,
            trusted_vectors,
            ancestry_ids=ancestry_ids,
            memory=self._document_memory,
        )
        if len(unique_docs) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "dedupe_exhausted_results"
            return []

        node.docs = unique_docs
        node.centroid = mean_embedding(unique_vectors)
        node.metrics.update(
            {
                "result_count": len(results),
                "trusted_count": len(trusted_docs),
                "unique_count": len(unique_docs),
            }
        )

        clusters = self.clusterer.cluster(unique_vectors)
        if not clusters:
            node.status = "stopped"
            node.stop_reason = "no_dense_clusters"
            self._remember_documents(node.node_id, unique_docs, unique_vectors)
            return []

        children: list[SearchNode] = []
        for cluster in clusters:
            cluster_docs = [unique_docs[index] for index in cluster.member_indexes]
            if len(cluster_docs) < self.config.limits.min_cluster_size:
                continue
            cluster_vectors = unique_vectors[cluster.member_indexes]
            child = self._make_child_node(tree, node, cluster_docs, cluster.centroid, cluster_vectors)
            if child is not None:
                children.append(child)

        self._remember_documents(node.node_id, unique_docs, unique_vectors)

        if not children:
            node.status = "stopped"
            node.stop_reason = "all_clusters_pruned"
            return []

        children.sort(key=lambda item: item.score, reverse=True)
        kept_children = children[: self.config.limits.max_children_per_node]
        pruned_children = children[self.config.limits.max_children_per_node :]

        for child in kept_children:
            tree.add_node(child)
            node.children.append(child.node_id)
            self._topic_memory.append(TopicMemoryItem(node_id=child.node_id, embedding=child.centroid))

        for child in pruned_children:
            child.status = "pruned"
            child.stop_reason = "per_node_pruning"
            tree.add_node(child)

        node.status = "expanded"
        return [child.node_id for child in kept_children]

    def _make_child_node(
        self,
        tree: SearchTree,
        parent: SearchNode,
        docs: list[SearchDocument],
        centroid: np.ndarray,
        vectors: np.ndarray,
    ) -> SearchNode | None:
        label, query, keywords = self.topic_builder.build(parent.query, docs)
        novelty = self._topic_novelty(parent.node_id, centroid)
        scope = max(0.0, cosine_similarity(centroid, self._root_query_vector))
        trust = float(np.mean([doc.trust_score for doc in docs]))
        support = len(docs) / max(self.config.limits.results_per_query, 1)
        size_score = min(len(docs) / max(self.config.limits.min_cluster_size, 1), 1.0)
        vector_consistency, drift = self.path_tracker.score(tree, parent.node_id, centroid)
        metrics = BranchMetrics(
            novelty=novelty,
            trust=trust,
            scope=scope,
            support=support,
            vector_consistency=vector_consistency,
            size_score=size_score,
            drift=drift,
        )
        decision = self.decision_model.evaluate(metrics)
        if novelty < self.config.similarity.novelty_threshold:
            return None
        if scope < self.config.similarity.scope_threshold:
            return None
        if not decision.allowed:
            return None

        return SearchNode(
            node_id=self._next_node_id(),
            query=query,
            depth=parent.depth + 1,
            parent_id=parent.node_id,
            score=decision.probability,
            cluster_label=label,
            docs=docs,
            centroid=centroid,
            metrics={
                "novelty": round(novelty, 4),
                "trust": round(trust, 4),
                "scope": round(scope, 4),
                "support": round(support, 4),
                "vector_consistency": round(vector_consistency, 4),
                "drift": round(drift, 4),
                "keyword_count": len(keywords),
                "cluster_size": len(vectors),
            },
        )

    def _topic_novelty(self, parent_node_id: str, centroid: np.ndarray) -> float:
        if not self._topic_memory:
            return 1.0
        max_similarity = 0.0
        for item in self._topic_memory:
            if item.node_id == parent_node_id:
                continue
            max_similarity = max(max_similarity, cosine_similarity(centroid, item.embedding))
        return max(0.0, 1.0 - max_similarity)

    def _remember_documents(
        self,
        node_id: str,
        docs: list[SearchDocument],
        embeddings: np.ndarray,
    ) -> None:
        for doc, embedding in zip(docs, embeddings):
            self._document_memory.append(
                DocumentMemoryItem(node_id=node_id, url=doc.url, embedding=embedding)
            )

    def _prune_frontier(self, tree: SearchTree, frontier: list[str]) -> list[str]:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for node_id in frontier:
            if node_id in seen:
                continue
            seen.add(node_id)
            unique_ids.append(node_id)

        unique_ids.sort(key=lambda node_id: tree.nodes[node_id].score, reverse=True)
        kept = unique_ids[: self.config.limits.frontier_width]
        dropped = unique_ids[self.config.limits.frontier_width :]
        for node_id in dropped:
            node = tree.nodes[node_id]
            if node.status == "pending":
                node.status = "pruned"
                node.stop_reason = "frontier_pruning"
        return kept

    def _next_node_id(self) -> str:
        return f"node-{next(self._id_counter)}"


def build_default_engine(config: EngineConfig | None = None) -> SearchTreeEngine:
    config = config or EngineConfig()
    embedder = SentenceTransformerEmbedder(config.transformer_model)
    provider = DuckDuckGoHtmlProvider()
    return SearchTreeEngine(provider=provider, embedder=embedder, config=config)
