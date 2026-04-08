from __future__ import annotations

import json
from dataclasses import replace
from itertools import count

import numpy as np

from .clustering import DensityClusterer
from .config import EngineConfig
from .contradictions import annotate_contradictions
from .embedding import SentenceTransformerEmbedder, TextEmbedder
from .falsehood import ClaimFalsehoodScorer
from .providers import MultiSearchProvider, SearchProvider
from .query_analysis import QueryAnalysis, analyze_query, rank_document_entities
from .retrieval import HybridRetrievalRanker, PageContentFetcher, passage_signature, tokenize_text
from .scoring import BranchDecisionModel, VectorPathTracker
from .topics import KeywordTopicBuilder
from .theme_tokens import ThemeTokenInducer
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
from .utils import cosine_similarity, is_at_or_before_as_of, mean_embedding, normalize_datetime_value
from .utils import jaccard_similarity


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
            if self._is_local_duplicate(doc, kept_docs):
                continue
            if self._is_global_duplicate(doc, ancestry_ids, memory):
                continue
            kept_docs.append(doc)
        return kept_docs

    def _is_local_duplicate(self, doc: SearchDocument, kept_docs: list[SearchDocument]) -> bool:
        vector = doc.embedding
        if vector is None:
            return False
        doc_signature = passage_signature(doc.passages[0].text if doc.passages else doc.text)
        doc_tokens = set(tokenize_text(doc_signature))
        return any(
            cosine_similarity(vector, other.embedding) >= self._threshold
            or self._signature_duplicate(doc_tokens, passage_signature(other.passages[0].text if other.passages else other.text))
            for other in kept_docs
        )

    def _is_global_duplicate(
        self,
        doc: SearchDocument,
        ancestry_ids: set[str],
        memory: list[DocumentMemoryItem],
    ) -> bool:
        vector = doc.embedding
        if vector is None:
            return False
        doc_tokens = set(tokenize_text(passage_signature(doc.passages[0].text if doc.passages else doc.text)))
        for item in memory:
            if item.node_id in ancestry_ids:
                continue
            if cosine_similarity(vector, item.embedding) >= self._threshold:
                return True
            if self._signature_duplicate(doc_tokens, item.passage_signature):
                return True
        return False

    def _signature_duplicate(self, left_tokens: set[str], right_signature: str) -> bool:
        if not left_tokens or not right_signature:
            return False
        right_tokens = set(tokenize_text(right_signature))
        return bool(right_tokens) and jaccard_similarity(left_tokens, right_tokens) >= 0.88


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
        self.theme_token_inducer = ThemeTokenInducer(self.config.theme_tokens)
        self.path_tracker = VectorPathTracker(self.config.scoring)
        self.decision_model = BranchDecisionModel(self.config.scoring)
        self.deduper = SimilarityDeduper(self.config.similarity.dedupe_threshold)
        self.retrieval_ranker = HybridRetrievalRanker(
            lexical_weight=self.config.retrieval.lexical_weight,
            dense_weight=self.config.retrieval.dense_weight,
            provider_weight=self.config.retrieval.provider_weight,
            domain_weight=self.config.retrieval.domain_weight,
            freshness_weight=self.config.retrieval.freshness_weight,
            entity_weight=self.config.retrieval.entity_weight,
            max_passages=self.config.retrieval.max_passages_per_doc,
            passage_dedupe_threshold=self.config.retrieval.passage_dedupe_threshold,
        )
        self._id_counter = count()
        self._document_memory: list[DocumentMemoryItem] = []
        self._topic_memory: list[TopicMemoryItem] = []
        self._root_query_vector: np.ndarray | None = None
        self._query_analysis: QueryAnalysis | None = None
        self._request_as_of: str | None = None
        self._asset_condition_label: str | None = None
        self._asset_condition_vector: np.ndarray | None = None

    def run(self, request: str | SearchRequest) -> SearchTree:
        search_request = self._coerce_request(request)
        effective_config = self._config_for_request(search_request)
        original_config = self.config

        self.config = effective_config
        try:
            self._reset_run_state()
            root_embedding = self.embedder.embed([search_request.query])
            self._root_query_vector = root_embedding[0]
            self._query_analysis = analyze_query(search_request.query)
            self._request_as_of = str(search_request.metadata.get("as_of") or "").strip() or None
            self._asset_condition_label, self._asset_condition_vector = self._asset_condition_for_query(search_request.query)

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
                        "intent": self._query_analysis.intent,
                        "provider_pack": self._query_analysis.domain_pack,
                        "as_of": self._request_as_of or "",
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

        enriched_docs = self._enrich_documents(results)
        filtered_docs = self._apply_as_of_cutoff(enriched_docs)
        context.trusted_document_ids = [doc.document_id for doc in filtered_docs]
        self._log_context_event(tree, context, "as_of_cutoff", {"count": len(filtered_docs)})
        if len(filtered_docs) < self.config.limits.min_results:
            node.status = "stopped"
            node.stop_reason = "as_of_cutoff_exhausted_results"
            context.stop_reason = node.stop_reason
            self._log_context_event(tree, context, "expand_stopped", {"reason": node.stop_reason})
            return []
        embedded_docs = self._embed_documents(filtered_docs)
        prepared_docs = self._prepare_documents(node.query, embedded_docs)
        trusted_docs = self.trust_scorer.assess_documents(prepared_docs, reference_time=self._request_as_of)
        trusted_docs = self.falsehood_scorer.assess_documents(trusted_docs)
        trusted_docs = self._rerank_documents(node.query, trusted_docs)
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
        contradiction_summary = annotate_contradictions(unique_docs)
        node.docs = unique_docs
        asset_label, asset_vector = self._asset_condition_for_query(node.query)
        node.theme_tokens = self.theme_token_inducer.induce(
            unique_docs,
            asset_vector=asset_vector,
            asset_label=asset_label,
        )
        self.topic_builder.decode_theme_tokens(node.query, unique_docs, node.theme_tokens)
        node.centroid = mean_embedding(unique_vectors)
        node.metrics.update(
            {
                "result_count": len(results),
                "trusted_count": len(trusted_docs),
                "unique_count": len(unique_docs),
                "contradiction_clusters": contradiction_summary["conflicting_cluster_count"],
                "theme_token_count": len(node.theme_tokens),
                "theme_edge_count": sum(len(token.outgoing_edges) for token in node.theme_tokens),
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

        children.sort(
            key=lambda item: (
                item.score,
                float(item.metrics.get("evidence_coverage", 0.0)),
                float(item.metrics.get("source_diversity", 0.0)),
            ),
            reverse=True,
        )
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
        asset_label, asset_vector = self._asset_condition_for_query(parent.query)
        theme_tokens = self.theme_token_inducer.induce(
            docs,
            asset_vector=asset_vector,
            asset_label=asset_label,
        )
        theme_topic = self.topic_builder.build_from_theme_graph(parent.query, docs, theme_tokens)
        topic = branch.topic if theme_topic is None else theme_topic
        if theme_topic is not None:
            node_metrics["keyword_count"] = len(theme_topic.keywords)
            node_metrics["evidence_count"] = len(theme_topic.evidence_document_ids)
        return SearchNode(
            node_id=self._next_node_id(),
            query=topic.query,
            depth=parent.depth + 1,
            parent_id=parent.node_id,
            score=branch.decision.probability,
            cluster_label=topic.label,
            docs=docs,
            centroid=centroid,
            metrics=node_metrics,
            topic=topic,
            theme_tokens=theme_tokens,
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
                    passage_signature=passage_signature(doc.passages[0].text if doc.passages else doc.text),
                )
            )

    def _embed_documents(self, docs: list[SearchDocument]) -> list[SearchDocument]:
        vectors = self.embedder.embed([doc.text for doc in docs])
        for doc, vector in zip(docs, vectors):
            doc.embedding = vector
        return docs

    def _enrich_documents(self, docs: list[SearchDocument]) -> list[SearchDocument]:
        if not self.config.retrieval.enable_page_fetch:
            return docs
        enrich_documents = getattr(self.provider, "enrich_documents", None)
        if not callable(enrich_documents):
            return docs
        return enrich_documents(docs, self.config.retrieval.fetch_top_k)

    def _prepare_documents(self, query: str, docs: list[SearchDocument]) -> list[SearchDocument]:
        analysis = self._analysis_for_query(query)
        ranked_entities = rank_document_entities(analysis, [doc.text for doc in docs])
        for doc in docs:
            profile = self.trust_scorer.resolve_source_profile(doc.source or doc.url)
            doc.source_profile = profile
            doc.metadata["domain_score"] = profile.domain_score
            doc.metadata["query_intent"] = analysis.intent
            doc.metadata["domain_pack"] = analysis.domain_pack
            doc.metadata["entity_hints"] = ranked_entities
            doc.metadata["ticker_hints"] = analysis.ticker_hints
        return self._rerank_documents(query, docs, entity_hints=ranked_entities, freshness_required=analysis.freshness_required)

    def _rerank_documents(
        self,
        query: str,
        docs: list[SearchDocument],
        *,
        entity_hints: list[str] | None = None,
        freshness_required: bool | None = None,
    ) -> list[SearchDocument]:
        if self._root_query_vector is None:
            return docs
        query_vector = self.embedder.embed([query])[0]
        analysis = self._analysis_for_query(query)
        hints = list(dict.fromkeys([*(entity_hints or []), *analysis.entity_hints]))
        return self.retrieval_ranker.rank(
            query=query,
            query_vector=query_vector,
            docs=docs,
            entity_hints=hints,
            freshness_required=analysis.freshness_required if freshness_required is None else freshness_required,
            as_of=self._request_as_of,
        )

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
        source_diversity = len({doc.source for doc in docs}) / max(len(docs), 1)
        evidence_coverage = float(np.mean([doc.passages[0].score if doc.passages else 0.0 for doc in docs]))
        falsehood_penalty = float(np.mean([doc.claim_falsehood_score for doc in docs]))
        freshness = float(np.mean([doc.freshness_score for doc in docs]))
        entity_alignment = float(np.mean([doc.entity_score for doc in docs]))
        contradiction_penalty = float(np.mean([float(doc.metadata.get("contradiction_penalty", 0.0)) for doc in docs]))
        metrics = BranchMetrics(
            novelty=novelty,
            trust=trust,
            scope=scope,
            support=support,
            vector_consistency=vector_consistency,
            size_score=size_score,
            drift=drift,
            source_diversity=source_diversity,
            evidence_coverage=evidence_coverage,
            falsehood_penalty=falsehood_penalty,
            freshness=freshness,
            entity_alignment=entity_alignment,
            contradiction_penalty=contradiction_penalty,
        )
        decision = self.decision_model.evaluate(metrics)
        return BranchState(topic=topic, metrics=metrics, decision=decision)

    def _embedding_matrix(self, docs: list[SearchDocument]) -> np.ndarray:
        vectors = [doc.embedding for doc in docs if doc.embedding is not None]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def _analysis_for_query(self, query: str) -> QueryAnalysis:
        analysis = analyze_query(query)
        if self._query_analysis is None:
            return analysis
        merged_entities = list(dict.fromkeys([*analysis.entity_hints, *self._query_analysis.entity_hints]))
        return QueryAnalysis(
            query=analysis.query,
            normalized_query=analysis.normalized_query,
            tokens=analysis.tokens,
            intent=analysis.intent,
            domain_pack=analysis.domain_pack,
            freshness_required=analysis.freshness_required or self._query_analysis.freshness_required,
            troubleshooting=analysis.troubleshooting,
            comparative=analysis.comparative,
            navigational=analysis.navigational,
            market_news=analysis.market_news or self._query_analysis.market_news,
            entity_hints=merged_entities,
            ticker_hints=list(dict.fromkeys([*analysis.ticker_hints, *self._query_analysis.ticker_hints])),
        )

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
        self._query_analysis = None
        self._request_as_of = None
        self._asset_condition_label = None
        self._asset_condition_vector = None
        self.trust_scorer.reset()

    def _next_node_id(self) -> str:
        return f"node-{next(self._id_counter)}"

    def _apply_as_of_cutoff(self, docs: list[SearchDocument]) -> list[SearchDocument]:
        if not self._request_as_of:
            return docs
        kept_docs: list[SearchDocument] = []
        for doc in docs:
            if doc.published_at:
                normalized, precision = normalize_datetime_value(doc.published_at)
                if normalized is not None:
                    doc.published_at = normalized
                    doc.published_at_precision = precision
            doc.metadata["published_at_precision"] = doc.published_at_precision
            doc.metadata["as_of"] = self._request_as_of
            if is_at_or_before_as_of(
                doc.published_at,
                self._request_as_of,
                published_precision=doc.published_at_precision,
            ):
                kept_docs.append(doc)
        return kept_docs

    def _asset_condition_for_query(self, query: str) -> tuple[str, np.ndarray | None]:
        analysis = self._analysis_for_query(query)
        if self._asset_condition_label and self._asset_condition_vector is not None:
            return self._asset_condition_label, self._asset_condition_vector
        parts: list[str] = []
        parts.extend(analysis.ticker_hints)
        parts.extend(analysis.entity_hints[:4])
        if not parts:
            parts.append(query)
        asset_label = " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))
        if not asset_label:
            return "", None
        asset_vector = self.embedder.embed([asset_label])[0]
        return asset_label, asset_vector


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
    provider = MultiSearchProvider(
        fetch_page_content=config.retrieval.enable_page_fetch,
        provider_pack=config.retrieval.provider_pack,
        fetcher=PageContentFetcher(
            timeout=10.0,
            max_concurrency=config.retrieval.async_concurrency,
            per_host_delay_seconds=config.retrieval.per_host_delay_seconds,
            cache_dir=config.retrieval.cache_dir,
            enable_cache=config.retrieval.enable_cache,
            cache_ttl_hours=config.retrieval.cache_ttl_hours,
        ) if config.retrieval.enable_page_fetch else None,
    )
    return SearchTreeEngine(provider=provider, embedder=embedder, config=config)
