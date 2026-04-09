from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import replace
from itertools import count

import numpy as np

from .config import EngineConfig
from .contradictions import annotate_contradictions
from .embedding import SentenceTransformerEmbedder, TextEmbedder
from .falsehood import ClaimFalsehoodScorer
from .providers import MultiSearchProvider, SearchProvider
from .query_analysis import QueryAnalysis, analyze_query, rank_document_entities
from .retrieval import HybridRetrievalRanker, PageContentFetcher, passage_signature, tokenize_text
from .scoring import ExpansionDecisionModel, VectorPathTracker
from .topics import KeywordTopicBuilder
from .theme_tokens import ThemeTokenInducer
from .trust import SourceTrustScorer
from .types import (
    DocumentMemoryItem,
    ExpansionMetrics,
    ExpansionState,
    ExpansionContext,
    SearchDocument,
    SearchRun,
    SearchRequest,
    SearchTopic,
    TopicProposal,
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
            if item.topic_id in ancestry_ids:
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


class RecursiveSearchEngine:
    def __init__(
        self,
        provider: SearchProvider,
        embedder: TextEmbedder,
        config: EngineConfig | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.provider = provider
        self.embedder = embedder
        self.trust_scorer = SourceTrustScorer(self.config.trust, embedder)
        self.falsehood_scorer = ClaimFalsehoodScorer(self.config.falsehood)
        self.topic_builder = KeywordTopicBuilder(embedder, self.config.topic_reranker)
        self.theme_token_inducer = ThemeTokenInducer(self.config.theme_tokens)
        self.path_tracker = VectorPathTracker(self.config.scoring)
        self.decision_model = ExpansionDecisionModel(self.config.scoring)
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

    def run(self, request: str | SearchRequest) -> SearchRun:
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

            root = SearchTopic(
                topic_id=self._next_topic_id(),
                query=search_request.query,
                depth=0,
                centroid=self._root_query_vector,
                status="pending",
                score=1.0,
            )
            run = SearchRun(root_topic_id=root.topic_id, request=search_request)
            run.add_topic(root)
            run.log_event(
                ExpansionContext(
                    topic_id=root.topic_id,
                    query=root.query,
                    depth=root.depth,
                ).record_event(
                    "search_started",
                    {
                        "max_depth": self.config.limits.max_depth,
                        "max_total_topics": self.config.limits.max_total_topics,
                        "frontier_width": self.config.limits.frontier_width,
                        "results_per_query": self.config.limits.results_per_query,
                        "intent": self._query_analysis.intent,
                        "provider_pack": self._query_analysis.domain_pack,
                        "as_of": self._request_as_of or "",
                    },
                )
            )
            frontier = [root.topic_id]

            while frontier and len(run.topics) < self.config.limits.max_total_topics:
                frontier.sort(key=lambda topic_id: run.topics[topic_id].score, reverse=True)
                current_id = frontier.pop(0)
                current = run.topics[current_id]

                new_topics = self._expand_topic(run, current)
                frontier.extend(new_topics)
                frontier = self._prune_frontier(run, frontier)

            for pending_id in frontier:
                pending = run.topics[pending_id]
                if pending.status == "pending":
                    pending.status = "pruned"
                    pending.stop_reason = "global_topic_limit"

            return run
        finally:
            self.config = original_config

    def dumps(self, run: SearchRun) -> str:
        return json.dumps(run.to_dict(), ensure_ascii=False, indent=2)

    def _expand_topic(self, run: SearchRun, topic: SearchTopic) -> list[str]:
        context = ExpansionContext(topic_id=topic.topic_id, query=topic.query, depth=topic.depth)
        run.add_expansion(context)
        self._log_context_event(run, context, "expand_started")

        if topic.depth >= self.config.limits.max_depth:
            topic.status = "stopped"
            topic.stop_reason = "max_depth"
            context.stop_reason = topic.stop_reason
            self._log_context_event(run, context, "expand_stopped", {"reason": topic.stop_reason})
            return []

        seed_results = [doc.clone() for doc in self.provider.search(topic.query, self.config.limits.results_per_query)]
        if len(seed_results) < self.config.limits.min_results:
            topic.status = "stopped"
            topic.stop_reason = "too_few_results"
            context.stop_reason = topic.stop_reason
            self._log_context_event(run, context, "expand_stopped", {"reason": topic.stop_reason})
            return []
        analysis = self._analysis_for_query(topic.query)
        asset_label, asset_vector = self._asset_condition_for_query(topic.query)
        ancestry_ids = set(run.ancestry_topic_ids(topic.topic_id) + [topic.topic_id])
        all_results = seed_results
        previous_state: np.ndarray | None = None
        final_docs: list[SearchDocument] = []
        final_tokens = []
        final_candidates: list[ThemeToken] = []
        state_delta = 1.0
        trusted_count = 0
        termination_reason = "state_converged"
        max_state_iterations = max(1, self.config.theme_tokens.max_state_iterations)

        for refinement_index in range(max_state_iterations):
            context.raw_document_ids = [doc.document_id for doc in all_results]
            self._log_context_event(run, context, "provider_results", {"count": len(all_results), "step": refinement_index})
            processed_docs = self._process_documents(
                run,
                topic.query,
                all_results,
                analysis=analysis,
                context=context,
                ancestry_ids=ancestry_ids,
            )
            final_docs = processed_docs
            if len(final_docs) < self.config.limits.min_results:
                topic.status = "stopped"
                topic.stop_reason = "insufficient_evidence"
                context.stop_reason = topic.stop_reason
                self._log_context_event(run, context, "expand_stopped", {"reason": topic.stop_reason})
                return []

            topic.theme_tokens = self.theme_token_inducer.induce(
                final_docs,
                asset_vector=asset_vector,
                asset_label=asset_label,
                embedder=self.embedder,
            )
            self.topic_builder.decode_theme_tokens(topic.query, final_docs, topic.theme_tokens)
            state_delta = self.theme_token_inducer.state_delta(previous_state)
            previous_state = self.theme_token_inducer.last_state.copy()
            trusted_count = len(context.trusted_document_ids)
            final_candidates = self._rank_theme_candidates(topic.theme_tokens)
            self._log_context_event(
                run,
                context,
                "theme_state",
                {
                    "step": refinement_index,
                    "theme_count": len(topic.theme_tokens),
                    "state_delta": round(state_delta, 6),
                },
            )

            if refinement_index > 0 and state_delta <= self.config.theme_tokens.state_convergence_threshold:
                termination_reason = "state_converged"
                break
            if refinement_index >= max_state_iterations - 1:
                termination_reason = "max_state_iterations"
                break
            if not final_candidates:
                termination_reason = "no_theme_candidates"
                break

            follow_up_results = self._search_theme_candidates(final_candidates)
            if not follow_up_results:
                termination_reason = "no_follow_up_results"
                break
            all_results = self._merge_documents(all_results, follow_up_results)

        unique_vectors = self._embedding_matrix(final_docs)
        contradiction_summary = annotate_contradictions(final_docs)
        topic.docs = final_docs
        topic.centroid = mean_embedding(unique_vectors)
        topic.metrics.update(
            {
                "result_count": len(all_results),
                "trusted_count": trusted_count,
                "unique_count": len(final_docs),
                "contradiction_clusters": contradiction_summary["conflicting_cluster_count"],
                "theme_token_count": len(topic.theme_tokens),
                "theme_edge_count": sum(len(token.outgoing_edges) for token in topic.theme_tokens),
                "state_delta": round(state_delta, 6),
            }
        )
        self._remember_documents(topic.topic_id, final_docs)

        subtopics = self._build_subtopics_from_themes(run, topic, final_docs, final_candidates)
        if not subtopics:
            topic.status = "stopped"
            topic.stop_reason = termination_reason
            context.stop_reason = topic.stop_reason
            self._log_context_event(
                run,
                context,
                "expand_stopped",
                {
                    "reason": topic.stop_reason,
                    "state_delta": round(state_delta, 6),
                    "iterations": max_state_iterations,
                },
            )
            return []

        remaining_slots = max(self.config.limits.max_total_topics - len(run.topics) - 1, 0)
        keep_limit = min(self.config.limits.max_subtopics_per_topic, remaining_slots)
        kept_subtopics = subtopics[:keep_limit]
        pruned_subtopics = subtopics[keep_limit:]

        for child in kept_subtopics:
            run.add_topic(child)
            topic.child_topic_ids.append(child.topic_id)
            self._topic_memory.append(TopicMemoryItem(topic_id=child.topic_id, embedding=child.centroid))

        for child in pruned_subtopics:
            child.status = "pruned"
            child.stop_reason = "per_topic_pruning"

        topic.status = "expanded"
        self._log_context_event(
            run,
            context,
            "expand_completed",
            {"subtopics_kept": len(kept_subtopics), "subtopics_pruned": len(pruned_subtopics)},
        )
        return [child.topic_id for child in kept_subtopics]

    def _make_child_topic(
        self,
        run: SearchRun,
        parent: SearchTopic,
        docs: list[SearchDocument],
        centroid: np.ndarray,
        *,
        proposal: TopicProposal | None = None,
        score: float | None = None,
        theme_id: str | None = None,
    ) -> SearchTopic | None:
        expansion = self._evaluate_expansion(run, parent, docs, centroid, proposal=proposal)
        if expansion.metrics.novelty < self.config.similarity.novelty_threshold:
            return None

        topic_metrics = expansion.to_topic_metrics()
        topic_metrics["evidence_count"] = len(docs)
        if theme_id:
            topic_metrics["theme_id"] = theme_id
        next_topic = expansion.topic
        return SearchTopic(
            topic_id=self._next_topic_id(),
            query=next_topic.query,
            depth=parent.depth + 1,
            parent_topic_id=parent.topic_id,
            score=expansion.decision.probability if score is None else score,
            cluster_label=next_topic.label,
            docs=docs,
            centroid=centroid,
            metrics=topic_metrics,
            topic=next_topic,
            theme_tokens=[],
        )

    def _topic_novelty(self, parent_topic_id: str, centroid: np.ndarray) -> float:
        if not self._topic_memory:
            return 1.0
        max_similarity = 0.0
        for item in self._topic_memory:
            if item.topic_id == parent_topic_id:
                continue
            max_similarity = max(max_similarity, cosine_similarity(centroid, item.embedding))
        return max(0.0, 1.0 - max_similarity)

    def _process_documents(
        self,
        run: SearchRun,
        query: str,
        docs: list[SearchDocument],
        *,
        analysis: QueryAnalysis,
        context: ExpansionContext,
        ancestry_ids: set[str],
    ) -> list[SearchDocument]:
        enriched_docs = self._enrich_documents([doc.clone() for doc in docs])
        filtered_docs = self._apply_as_of_cutoff(enriched_docs)
        self._log_context_event(run, context, "as_of_cutoff", {"count": len(filtered_docs)})
        embedded_docs = self._embed_documents(filtered_docs)
        annotated_docs, ranked_entities = self._annotate_documents(embedded_docs, analysis)
        trusted_docs = self._filter_documents(annotated_docs)
        trusted_docs = self._rank_documents(
            query,
            trusted_docs,
            entity_hints=ranked_entities,
            freshness_required=analysis.freshness_required,
        )
        context.trusted_document_ids = [doc.document_id for doc in trusted_docs]
        self._log_context_event(run, context, "trust_filter", {"count": len(trusted_docs)})
        unique_docs = self.deduper.dedupe(
            trusted_docs,
            ancestry_ids=ancestry_ids,
            memory=self._document_memory,
        )
        context.unique_document_ids = [doc.document_id for doc in unique_docs]
        self._log_context_event(run, context, "dedupe", {"count": len(unique_docs)})
        return unique_docs

    def _rank_theme_candidates(self, theme_tokens: list) -> list:
        ranked = [token for token in theme_tokens if token.query.strip() and token.score > 0.0]
        ranked.sort(
            key=lambda token: (
                token.score + (0.12 * self._theme_support(token)),
                float(token.candidate_scores[0]["score"]) if token.candidate_scores else 0.0,
                token.base_score,
            ),
            reverse=True,
        )
        return ranked

    def _theme_support(self, token) -> float:
        corpus = " ".join([*token.top_keywords, *token.anchor_titles, *token.attention_titles]).lower()
        if not corpus:
            return 0.0
        support_terms = set(token.token_id.split())
        return float(sum(1 for term in support_terms if term in corpus))

    def _search_theme_candidates(self, theme_tokens: list) -> list[SearchDocument]:
        if not theme_tokens:
            return []
        topic_count = min(len(theme_tokens), self.config.limits.max_subtopics_per_topic)
        per_topic_budget = max(1, self.config.limits.results_per_query // max(topic_count, 1))
        results: list[SearchDocument] = []
        for token in theme_tokens[:topic_count]:
            results.extend(doc.clone() for doc in self.provider.search(token.query, per_topic_budget))
        return results

    def _merge_documents(
        self,
        base_docs: list[SearchDocument],
        new_docs: list[SearchDocument],
    ) -> list[SearchDocument]:
        merged: OrderedDict[str, SearchDocument] = OrderedDict()
        for doc in [*base_docs, *new_docs]:
            key = doc.document_id or doc.url
            if key not in merged:
                merged[key] = doc
        return list(merged.values())

    def _theme_docs(
        self,
        docs: list[SearchDocument],
        token,
    ) -> list[SearchDocument]:
        doc_by_id = {doc.document_id: doc for doc in docs}
        ordered_ids = list(dict.fromkeys([*token.attention_document_ids, *token.anchor_document_ids]))
        return [doc_by_id[document_id] for document_id in ordered_ids if document_id in doc_by_id]

    def _proposal_from_theme(self, token) -> TopicProposal:
        evidence_ids = list(dict.fromkeys([*token.attention_document_ids, *token.anchor_document_ids]))[:4]
        query_parts = [token.token_id, *token.top_keywords[:2]]
        query = " ".join(dict.fromkeys(part.strip() for part in query_parts if part.strip()))
        label_parts = [token.token_id, *token.top_keywords[:2]]
        label = ", ".join(dict.fromkeys(part.strip() for part in label_parts if part.strip()))
        return TopicProposal(
            label=label or token.token_id,
            query=query or token.token_id,
            keywords=list(dict.fromkeys([token.token_id, *token.top_keywords])),
            evidence_document_ids=evidence_ids,
            candidate_scores=list(token.candidate_scores),
        )

    def _build_subtopics_from_themes(
        self,
        run: SearchRun,
        parent: SearchTopic,
        docs: list[SearchDocument],
        theme_tokens: list,
    ) -> list[SearchTopic]:
        subtopics: list[SearchTopic] = []
        for token in theme_tokens:
            themed_docs = self._theme_docs(docs, token)
            if len(themed_docs) < max(2, self.config.limits.min_cluster_size - 1):
                continue
            centroid = mean_embedding(self._embedding_matrix(themed_docs))
            if centroid is None:
                continue
            child = self._make_child_topic(
                run,
                parent,
                themed_docs,
                centroid,
                proposal=self._proposal_from_theme(token),
                score=float(token.score + (0.12 * self._theme_support(token))),
                theme_id=token.token_id,
            )
            if child is not None:
                subtopics.append(child)
        subtopics.sort(
            key=lambda item: (
                item.score,
                float(item.metrics.get("evidence_coverage", 0.0)),
                float(item.metrics.get("source_diversity", 0.0)),
            ),
            reverse=True,
        )
        return subtopics

    def _remember_documents(
        self,
        topic_id: str,
        docs: list[SearchDocument],
    ) -> None:
        for doc in docs:
            if doc.embedding is None:
                continue
            self._document_memory.append(
                DocumentMemoryItem(
                    topic_id=topic_id,
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

    def _annotate_documents(
        self,
        docs: list[SearchDocument],
        analysis: QueryAnalysis,
    ) -> tuple[list[SearchDocument], list[str]]:
        ranked_entities = rank_document_entities(analysis, [doc.text for doc in docs])
        for doc in docs:
            profile = self.trust_scorer.resolve_source_profile(doc.source or doc.url)
            doc.source_profile = profile
            doc.metadata["domain_score"] = profile.domain_score
            doc.metadata["query_intent"] = analysis.intent
            doc.metadata["domain_pack"] = analysis.domain_pack
            doc.metadata["entity_hints"] = ranked_entities
            doc.metadata["ticker_hints"] = analysis.ticker_hints
        return docs, ranked_entities

    def _filter_documents(self, docs: list[SearchDocument]) -> list[SearchDocument]:
        trusted_docs = self.trust_scorer.assess_documents(docs, reference_time=self._request_as_of)
        return self.falsehood_scorer.assess_documents(trusted_docs)

    def _rank_documents(
        self,
        query: str,
        docs: list[SearchDocument],
        *,
        entity_hints: list[str] | None = None,
        freshness_required: bool | None = None,
    ) -> list[SearchDocument]:
        if not docs:
            return []
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

    def _evaluate_expansion(
        self,
        run: SearchRun,
        parent: SearchTopic,
        docs: list[SearchDocument],
        centroid: np.ndarray,
        *,
        proposal: TopicProposal | None = None,
    ) -> ExpansionState:
        topic = proposal if proposal is not None else self.topic_builder.build(parent.query, docs)
        novelty = self._topic_novelty(parent.topic_id, centroid)
        scope = max(0.0, cosine_similarity(centroid, self._root_query_vector))
        trust = float(np.mean([doc.trust_score for doc in docs]))
        support = len(docs) / max(self.config.limits.results_per_query, 1)
        size_score = min(len(docs) / max(self.config.limits.min_cluster_size, 1), 1.0)
        vector_consistency, drift = self.path_tracker.score(run, parent.topic_id, centroid)
        source_diversity = len({doc.source for doc in docs}) / max(len(docs), 1)
        evidence_coverage = float(np.mean([doc.passages[0].score if doc.passages else 0.0 for doc in docs]))
        falsehood_penalty = float(np.mean([doc.claim_falsehood_score for doc in docs]))
        freshness = float(np.mean([doc.freshness_score for doc in docs]))
        entity_alignment = float(np.mean([doc.entity_score for doc in docs]))
        contradiction_penalty = float(np.mean([float(doc.metadata.get("contradiction_penalty", 0.0)) for doc in docs]))
        metrics = ExpansionMetrics(
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
        return ExpansionState(topic=topic, metrics=metrics, decision=decision)

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
            max_total_topics=(
                request.max_total_topics
                if request.max_total_topics is not None
                else self.config.limits.max_total_topics
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
        run: SearchRun,
        context: ExpansionContext,
        event_type: str,
        payload: dict[str, int | float | str] | None = None,
    ) -> None:
        run.log_event(context.record_event(event_type, {} if payload is None else dict(payload)))

    def _prune_frontier(self, run: SearchRun, frontier: list[str]) -> list[str]:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for topic_id in frontier:
            if topic_id in seen:
                continue
            seen.add(topic_id)
            unique_ids.append(topic_id)

        unique_ids.sort(key=lambda topic_id: run.topics[topic_id].score, reverse=True)
        kept = unique_ids[: self.config.limits.frontier_width]
        dropped = unique_ids[self.config.limits.frontier_width :]
        for topic_id in dropped:
            topic = run.topics[topic_id]
            if topic.status == "pending":
                topic.status = "pruned"
                topic.stop_reason = "frontier_pruning"
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

    def _next_topic_id(self) -> str:
        return f"topic-{next(self._id_counter)}"

    def _next_node_id(self) -> str:
        return self._next_topic_id()

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

    def _expand_node(self, run: SearchRun, topic: SearchTopic) -> list[str]:
        return self._expand_topic(run, topic)

    def _make_child_node(
        self,
        run: SearchRun,
        parent: SearchTopic,
        docs: list[SearchDocument],
        centroid: np.ndarray,
    ) -> SearchTopic | None:
        return self._make_child_topic(run, parent, docs, centroid)

    def _evaluate_branch(
        self,
        run: SearchRun,
        parent: SearchTopic,
        docs: list[SearchDocument],
        centroid: np.ndarray,
    ) -> ExpansionState:
        return self._evaluate_expansion(run, parent, docs, centroid)


def build_default_engine(config: EngineConfig | None = None) -> RecursiveSearchEngine:
    config = config or EngineConfig()
    if config.trust_model_path is not None:
        config.trust.mlp_model_path = config.trust_model_path
    model_path = config.expansion_model_path or config.branch_model_path
    if model_path is not None:
        config.scoring.mlp_model_path = model_path
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
    return RecursiveSearchEngine(provider=provider, embedder=embedder, config=config)


SearchTreeEngine = RecursiveSearchEngine
