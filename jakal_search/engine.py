from __future__ import annotations

import json
from dataclasses import replace
from itertools import count

import numpy as np

from .clustering import DensityClusterer
from .config import EngineConfig
from .embedding import SentenceTransformerEmbedder, TextEmbedder
from .falsehood import ClaimFalsehoodScorer
from .providers import DuckDuckGoHtmlProvider, SearchProvider
from .scoring import BranchDecisionModel, VectorPathTracker
from .topics import KeywordTopicBuilder
from .trust import SourceTrustScorer
from .types import (
    BranchMetrics,
    BranchState,
    DocumentMemoryItem,
    ExpansionContext,
    SearchDocument,
    SearchNode,
    SearchRequest,
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
        ancestry_ids: set[str],
        memory: list[DocumentMemoryItem],
    ) -> list[SearchDocument]:
        kept_docs: list[SearchDocument] = []

        for doc in docs:
            vector = doc.embedding
            if vector is None or vector.size == 0:
                continue
            if self._is_local_duplicate(vector, kept_docs):
                continue
            if self._is_global_duplicate(vector, ancestry_ids, memory):
                continue
            kept_docs.append(doc)
        return kept_docs

    def _is_local_duplicate(self, vector: np.ndarray, kept_docs: list[SearchDocument]) -> bool:
        return any(cosine_similarity(vector, other.embedding) >= self._threshold for other in kept_docs)

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
        self.falsehood_scorer = ClaimFalsehoodScorer(self.config.falsehood)
        self.topic_builder = KeywordTopicBuilder(embedder, self.config.topic_reranker)
        self.path_tracker = VectorPathTracker(self.config.scoring)
        self.decision_model = BranchDecisionModel(self.config.scoring)
        self.deduper = SimilarityDeduper(self.config.similarity.dedupe_threshold)
        self._id_counter = count()
        self._document_memory: list[DocumentMemoryItem] = []
        self._topic_memory: list[TopicMemoryItem] = []
        self._root_query_vector: np.ndarray | None = None

    def run(self, request: str | SearchRequest) -> SearchTree:
        search_request = self._coerce_request(request)
        effective_config = self._config_for_request(search_request)
        original_config = self.config

        self.config = effective_config
        try:
            self._reset_run_state()
            root_embedding = self.embedder.embed([search_request.query])
            self._root_query_vector = root_embedding[0]

            root = SearchNode(
                node_id=self._next_node_id(),
                query=search_request.query,
                depth=0,
                centroid=self._root_query_vector,
                status="pending",
                score=1.0,
            )
            tree = SearchTree(root_id=root.node_id, request=search_request)
            tree.add_node(root)
            tree.log_event(
                ExpansionContext(
                    node_id=root.node_id,
                    query=root.query,
                    depth=root.depth,
                ).record_event(
                    "search_started",
                    {
                        "max_depth": self.config.limits.max_depth,
                        "max_total_nodes": self.config.limits.max_total_nodes,
                        "frontier_width": self.config.limits.frontier_width,
                        "results_per_query": self.config.limits.results_per_query,
                    },
                )
            )
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
        finally:
            self.config = original_config

    def dumps(self, tree: SearchTree) -> str:
        return json.dumps(tree.to_dict(), ensure_ascii=False, indent=2)

    def _expand_node(self, tree: SearchTree, node: SearchNode) -> list[str]:
        context = ExpansionContext(node_id=node.node_id, query=node.query, depth=node.depth)
        tree.add_expansion(context)
        self._log_context_event(tree, context, "expand_started")

        if node.depth >= self.config.limits.max_depth:
            node.status = "stopped"
            node.stop_reason = "max_depth"
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
            return []

        results = [doc.clone() for doc in self.provider.search(node.query, self.config.limits.results_per_query)]
        context.raw_document_ids = [doc.document_id for doc in results]
        self._log_context_event(tree, context, "provider_results", {"count": len(results)})
        if len(results) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "too_few_results"
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
            return []

        embedded_docs = self._embed_documents(results)
        trusted_docs = self.trust_scorer.assess_documents(embedded_docs)
        trusted_docs = self.falsehood_scorer.assess_documents(trusted_docs)
        context.trusted_document_ids = [doc.document_id for doc in trusted_docs]
        self._log_context_event(tree, context, "trust_filter", {"count": len(trusted_docs)})
        if len(trusted_docs) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "trust_filter_exhausted_results"
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
            return []

        ancestry_ids = set(tree.ancestry_ids(node.node_id) + [node.node_id])
        unique_docs = self.deduper.dedupe(
            trusted_docs,
            ancestry_ids=ancestry_ids,
            memory=self._document_memory,
        )
        context.unique_document_ids = [doc.document_id for doc in unique_docs]
        self._log_context_event(tree, context, "dedupe", {"count": len(unique_docs)})
        if len(unique_docs) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "dedupe_exhausted_results"
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
            return []

        unique_vectors = self._embedding_matrix(unique_docs)
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
        context.cluster_ids = [int(cluster.cluster_id) for cluster in clusters]
        self._log_context_event(tree, context, "clustering", {"count": len(clusters)})
        if not clusters:
            node.status = "stopped"
            node.stop_reason = "no_dense_clusters"
            self._remember_documents(node.node_id, unique_docs)
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
            return []

        children: list[SearchNode] = []
        for cluster in clusters:
            cluster_docs = [unique_docs[index] for index in cluster.member_indexes]
            if len(cluster_docs) < self.config.limits.min_cluster_size:
                continue
            for doc in cluster_docs:
                doc.cluster_id = cluster.cluster_id
            child = self._make_child_node(tree, node, cluster_docs, cluster.centroid)
            if child is not None:
                children.append(child)

        self._remember_documents(node.node_id, unique_docs)

        if not children:
            node.status = "stopped"
            node.stop_reason = "all_clusters_pruned"
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
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
        self._log_context_event(
            tree,
            context,
            "expand_completed",
            {"children_kept": len(kept_children), "children_pruned": len(pruned_children)},
        )
        return [child.node_id for child in kept_children]

    def _make_child_node(
        self,
        tree: SearchTree,
        parent: SearchNode,
        docs: list[SearchDocument],
        centroid: np.ndarray,
    ) -> SearchNode | None:
        branch = self._evaluate_branch(tree, parent, docs, centroid)
        if branch.metrics.novelty < self.config.similarity.novelty_threshold:
            return None
        if branch.metrics.scope < self.config.similarity.scope_threshold:
            return None
        if not branch.decision.allowed:
            return None

        node_metrics = branch.to_node_metrics()
        node_metrics["cluster_size"] = len(docs)
        return SearchNode(
            node_id=self._next_node_id(),
            query=branch.topic.query,
            depth=parent.depth + 1,
            parent_id=parent.node_id,
            score=branch.decision.probability,
            cluster_label=branch.topic.label,
            docs=docs,
            centroid=centroid,
            metrics=node_metrics,
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
    ) -> None:
        for doc in docs:
            if doc.embedding is None:
                continue
            self._document_memory.append(
                DocumentMemoryItem(
                    node_id=node_id,
                    document_id=doc.document_id,
                    url=doc.url,
                    embedding=doc.embedding,
                )
            )

    def _embed_documents(self, docs: list[SearchDocument]) -> list[SearchDocument]:
        vectors = self.embedder.embed([doc.text for doc in docs])
        for doc, vector in zip(docs, vectors):
            doc.embedding = vector
        return docs

    def _evaluate_branch(
        self,
        tree: SearchTree,
        parent: SearchNode,
        docs: list[SearchDocument],
        centroid: np.ndarray,
    ) -> BranchState:
        topic = self.topic_builder.build(parent.query, docs)
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
        return BranchState(topic=topic, metrics=metrics, decision=decision)

    def _embedding_matrix(self, docs: list[SearchDocument]) -> np.ndarray:
        vectors = [doc.embedding for doc in docs if doc.embedding is not None]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def _coerce_request(self, request: str | SearchRequest) -> SearchRequest:
        if isinstance(request, SearchRequest):
            return request
        return SearchRequest(query=request)

    def _config_for_request(self, request: SearchRequest) -> EngineConfig:
        limits = replace(
            self.config.limits,
            max_depth=request.max_depth if request.max_depth is not None else self.config.limits.max_depth,
            max_total_nodes=(
                request.max_total_nodes
                if request.max_total_nodes is not None
                else self.config.limits.max_total_nodes
            ),
            frontier_width=(
                request.frontier_width
                if request.frontier_width is not None
                else self.config.limits.frontier_width
            ),
            results_per_query=(
                request.results_per_query
                if request.results_per_query is not None
                else self.config.limits.results_per_query
            ),
        )
        return replace(self.config, limits=limits)

    def _log_context_event(
        self,
        tree: SearchTree,
        context: ExpansionContext,
        event_type: str,
        payload: dict[str, int | float | str] | None = None,
    ) -> None:
        tree.log_event(context.record_event(event_type, {} if payload is None else dict(payload)))

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

    def _reset_run_state(self) -> None:
        self._id_counter = count()
        self._document_memory = []
        self._topic_memory = []
        self._root_query_vector = None
        self.trust_scorer.reset()

    def _next_node_id(self) -> str:
        return f"node-{next(self._id_counter)}"


def build_default_engine(config: EngineConfig | None = None) -> SearchTreeEngine:
    config = config or EngineConfig()
    if config.trust_model_path is not None:
        config.trust.mlp_model_path = config.trust_model_path
    if config.branch_model_path is not None:
        config.scoring.mlp_model_path = config.branch_model_path
    if config.topic_reranker_model_path is not None:
        config.topic_reranker.model_path = config.topic_reranker_model_path
    if config.falsehood_model_path is not None:
        config.falsehood.model_path = config.falsehood_model_path
    embedder = SentenceTransformerEmbedder(
        config.transformer_model,
        device=config.transformer_device,
    )
    provider = DuckDuckGoHtmlProvider()
    return SearchTreeEngine(provider=provider, embedder=embedder, config=config)
