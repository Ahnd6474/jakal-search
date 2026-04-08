from __future__ import annotations

import re
from collections import Counter
import itertools

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TopicRerankerConfig
from .embedding import TextEmbedder
from .reranking import TopicQueryReranker
from .types import SearchDocument, ThemeToken, TopicProposal
from .utils import cosine_similarity, mean_embedding

PHRASE_PATTERN = re.compile(r"\b[a-z][a-z0-9\-]{2,}(?:\s+[a-z][a-z0-9\-]{2,}){0,3}\b", re.IGNORECASE)
GENERIC_TOPIC_TOKENS = {"evidence", "explanation", "details", "detail", "overview", "guide", "steps", "context"}


class KeywordTopicBuilder:
    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        reranker_config: TopicRerankerConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._reranker_config = reranker_config or TopicRerankerConfig()
        self._reranker = None if embedder is None else TopicQueryReranker(self._reranker_config, embedder)

    def build(self, parent_query: str, docs: list[SearchDocument]) -> TopicProposal:
        theme_topic = self.build_from_theme_graph(parent_query, docs, self._theme_tokens_from_docs(docs))
        if theme_topic is not None:
            return theme_topic
        entity_hints = list(
            dict.fromkeys(
                hint
                for doc in docs
                for hint in list(doc.metadata.get("entity_hints") or [])
            )
        )
        texts = self._topic_texts(docs)
        if not texts:
            return TopicProposal(
                label=parent_query,
                query=parent_query,
                keywords=[],
                evidence_document_ids=[doc.document_id for doc in docs[:3]],
                candidate_scores=[],
            )

        keywords = self._extract_keywords(texts, parent_query, entity_hints)
        scored_candidates = self._score_keyword_candidates(parent_query, docs, keywords)
        reranked_keywords = [item["candidate"] for item in scored_candidates]
        reranked_keywords = self._diversify_keywords(parent_query, reranked_keywords)
        label_terms = reranked_keywords[:3]
        label = ", ".join(label_terms) if label_terms else docs[0].title[:60]
        query = self._build_query(parent_query, docs, reranked_keywords, entity_hints)
        return TopicProposal(
            label=label,
            query=query,
            keywords=reranked_keywords,
            evidence_document_ids=[doc.document_id for doc in docs[:3]],
            candidate_scores=scored_candidates[:8],
        )

    def build_from_theme_graph(
        self,
        parent_query: str,
        docs: list[SearchDocument],
        theme_tokens: list[ThemeToken],
    ) -> TopicProposal | None:
        decoded_topics = self.decode_theme_tokens(parent_query, docs, theme_tokens)
        if not decoded_topics:
            return None
        return max(
            decoded_topics,
            key=lambda topic: (
                len(topic.evidence_document_ids),
                len(topic.keywords),
                float(topic.candidate_scores[0]["score"]) if topic.candidate_scores else 0.0,
            ),
        )

    def decode_theme_tokens(
        self,
        parent_query: str,
        docs: list[SearchDocument],
        theme_tokens: list[ThemeToken],
    ) -> list[TopicProposal]:
        proposals: list[TopicProposal] = []
        if not docs or not theme_tokens:
            return proposals
        doc_by_id = {doc.document_id: doc for doc in docs}
        entity_hints = list(
            dict.fromkeys(
                hint
                for doc in docs
                for hint in list(doc.metadata.get("entity_hints") or [])
            )
        )
        for token in theme_tokens:
            proposal = self._build_for_theme(parent_query, docs, doc_by_id, token, entity_hints)
            if proposal is None:
                continue
            token.label = proposal.label
            token.query = proposal.query
            token.top_keywords = list(proposal.keywords[:4])
            token.candidate_scores = proposal.candidate_scores[:8]
            proposals.append(proposal)
        return proposals

    def _topic_texts(self, docs: list[SearchDocument]) -> list[str]:
        texts: list[str] = []
        for doc in docs:
            if doc.title:
                texts.extend([doc.title, doc.title])
            if doc.content:
                texts.append(doc.content)
        return texts

    def _theme_tokens_from_docs(self, docs: list[SearchDocument]) -> list[ThemeToken]:
        return []

    def _build_for_theme(
        self,
        parent_query: str,
        docs: list[SearchDocument],
        doc_by_id: dict[str, SearchDocument],
        token: ThemeToken,
        entity_hints: list[str],
    ) -> TopicProposal | None:
        if token.topic_vector is None:
            return None
        candidate_docs = self._theme_docs(doc_by_id, token)
        texts = self._topic_texts(candidate_docs)
        if not texts:
            return None
        keywords = self._extract_keywords(texts, parent_query, entity_hints)
        scored_candidates = self._score_theme_candidates(parent_query, token, candidate_docs, keywords)
        reranked_keywords = [item["candidate"] for item in scored_candidates]
        reranked_keywords = self._diversify_keywords(parent_query, reranked_keywords)
        if not reranked_keywords:
            reranked_keywords = list(token.top_keywords)
        label_terms = reranked_keywords[:3]
        label = ", ".join(label_terms) if label_terms else token.token_id
        query = self._build_query(parent_query, candidate_docs, reranked_keywords, entity_hints)
        return TopicProposal(
            label=label,
            query=query,
            keywords=reranked_keywords,
            evidence_document_ids=[doc.document_id for doc in candidate_docs[:4]],
            candidate_scores=scored_candidates[:8],
        )

    def _theme_docs(
        self,
        doc_by_id: dict[str, SearchDocument],
        token: ThemeToken,
    ) -> list[SearchDocument]:
        ordered_ids = list(dict.fromkeys([*token.attention_document_ids, *token.anchor_document_ids]))
        return [doc_by_id[document_id] for document_id in ordered_ids if document_id in doc_by_id]

    def _extract_keywords(self, texts: list[str], parent_query: str, entity_hints: list[str]) -> list[str]:
        phrase_candidates = self._extract_phrase_candidates(texts, parent_query)
        phrase_candidates = list(dict.fromkeys([*entity_hints, *phrase_candidates]))
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_features=96,
                min_df=1,
            )
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return phrase_candidates[:8]

        scores = np.asarray(matrix.mean(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        parent_tokens = set(parent_query.lower().split())
        weighted: list[tuple[float, str]] = []
        phrase_support = Counter(phrase_candidates)
        for feature, score in zip(features, scores):
            feature_tokens = set(feature.lower().split())
            if not feature_tokens or feature_tokens.issubset(parent_tokens):
                continue
            support = phrase_support.get(feature, 0)
            weighted.append((float(score) + (0.12 * support), feature))

        ranked = [feature for _, feature in sorted(weighted, key=lambda item: (item[0], len(item[1])), reverse=True)]
        merged = list(dict.fromkeys([*phrase_candidates, *ranked]))
        if self._should_use_transformer_keywords():
            merged = self._rank_candidates_with_transformer(parent_query, texts, merged)
        return merged[:10]

    def _extract_phrase_candidates(self, texts: list[str], parent_query: str) -> list[str]:
        parent_tokens = set(parent_query.lower().split())
        counts: Counter[str] = Counter()
        for text in texts:
            for match in PHRASE_PATTERN.findall(text):
                phrase = " ".join(match.lower().split())
                tokens = phrase.split()
                if len(tokens) > 4:
                    continue
                if set(tokens).issubset(parent_tokens):
                    continue
                if len(tokens) == 1 and len(tokens[0]) < 5:
                    continue
                if tokens[0] in GENERIC_TOPIC_TOKENS:
                    continue
                if len(parent_tokens & set(tokens)) >= max(2, len(tokens) - 1):
                    continue
                counts[phrase] += 1
        return [
            phrase
            for phrase, _ in counts.most_common(12)
            if phrase and not phrase.startswith(("http", "www"))
        ]

    def _score_keyword_candidates(
        self,
        parent_query: str,
        docs: list[SearchDocument],
        keywords: list[str],
    ) -> list[dict[str, float | str]]:
        if not keywords:
            return []
        if self._reranker is None or not self._reranker_config.enabled:
            return [{"candidate": keyword, "score": float(len(keywords) - index)} for index, keyword in enumerate(keywords)]
        candidates = keywords[: self._reranker_config.candidate_pool_size]
        return self._reranker.score_candidates(parent_query=parent_query, docs=docs, candidates=candidates)

    def _score_theme_candidates(
        self,
        parent_query: str,
        token: ThemeToken,
        docs: list[SearchDocument],
        keywords: list[str],
    ) -> list[dict[str, float | str]]:
        if not keywords:
            return []
        if self._embedder is None or token.topic_vector is None:
            return self._score_keyword_candidates(parent_query, docs, keywords)
        candidate_embeddings = self._embedder.embed(keywords)
        topic_vector = token.topic_vector
        text_blob = " ".join(doc.text.lower() for doc in docs)
        parent_tokens = set(parent_query.lower().split())
        scored: list[dict[str, float | str]] = []
        for candidate, vector in zip(keywords, candidate_embeddings):
            support = sum(1 for doc in docs if candidate in doc.text.lower()) / max(len(docs), 1)
            overlap = len(set(candidate.split()) & parent_tokens) / max(len(candidate.split()), 1)
            generic_penalty = 0.2 if any(term in GENERIC_TOPIC_TOKENS for term in candidate.split()) else 0.0
            lexical_support = 1.0 if candidate in text_blob else support
            score = (
                (0.62 * cosine_similarity(topic_vector, vector))
                + (0.18 * lexical_support)
                + (0.10 * support)
                + (0.10 * (1.0 - overlap))
                - generic_penalty
            )
            scored.append({"candidate": candidate, "score": float(score)})
        scored.sort(key=lambda item: (float(item["score"]), len(str(item["candidate"]))), reverse=True)
        return scored

    def _diversify_keywords(self, parent_query: str, keywords: list[str]) -> list[str]:
        if not keywords:
            return []
        if self._should_use_transformer_keywords():
            return self._mmr_diversify(parent_query, keywords)
        diversified: list[str] = []
        parent_tokens = set(parent_query.lower().split())
        for candidate in keywords:
            candidate_tokens = set(candidate.split())
            if candidate_tokens.issubset(parent_tokens):
                continue
            if any(candidate in existing or existing in candidate for existing in diversified):
                continue
            if any(len(candidate_tokens & set(existing.split())) >= max(1, min(len(candidate_tokens), 2)) for existing in diversified):
                continue
            diversified.append(candidate)
            if len(diversified) >= 6:
                break
        return diversified or keywords[:6]

    def _should_use_transformer_keywords(self) -> bool:
        return bool(self._embedder is not None and getattr(self._embedder, "using_transformer", False))

    def _rank_candidates_with_transformer(
        self,
        parent_query: str,
        texts: list[str],
        candidates: list[str],
    ) -> list[str]:
        if self._embedder is None or not candidates:
            return candidates
        doc_vectors = self._embedder.embed(texts[: min(len(texts), 8)])
        centroid = mean_embedding(np.asarray(doc_vectors, dtype=np.float32))
        vectors = self._embedder.embed([parent_query, *candidates])
        parent_vector = vectors[0]
        candidate_vectors = vectors[1:]
        scored: list[tuple[float, str]] = []
        for candidate, vector in zip(candidates, candidate_vectors):
            query_alignment = cosine_similarity(parent_vector, vector)
            centroid_alignment = 0.0 if centroid is None else cosine_similarity(centroid, vector)
            overlap = len(set(candidate.split()) & set(parent_query.lower().split())) / max(len(candidate.split()), 1)
            generic_penalty = 0.2 if any(token in GENERIC_TOPIC_TOKENS for token in candidate.split()) else 0.0
            score = (0.3 * query_alignment) + (0.55 * centroid_alignment) + (0.15 * (1.0 - overlap)) - generic_penalty
            scored.append((score, candidate))
        scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return [candidate for _, candidate in scored]

    def _mmr_diversify(self, parent_query: str, keywords: list[str]) -> list[str]:
        if self._embedder is None:
            return keywords[:6]
        vectors = self._embedder.embed([parent_query, *keywords])
        parent_vector = vectors[0]
        keyword_vectors = vectors[1:]
        remaining = list(zip(keywords, keyword_vectors))
        selected: list[tuple[str, np.ndarray]] = []
        while remaining and len(selected) < 6:
            best_index = 0
            best_score = float("-inf")
            for index, (candidate, vector) in enumerate(remaining):
                relevance = cosine_similarity(parent_vector, vector)
                redundancy = max((cosine_similarity(vector, chosen_vector) for _, chosen_vector in selected), default=0.0)
                score = (0.72 * relevance) - (0.28 * redundancy)
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(remaining.pop(best_index))
        return [candidate for candidate, _ in selected]

    def _build_query(
        self,
        parent_query: str,
        docs: list[SearchDocument],
        keywords: list[str],
        entity_hints: list[str],
    ) -> str:
        if not keywords:
            return parent_query
        trusted_sources = [doc.source for doc in docs if doc.trust_score >= 0.7]
        dominant_source = Counter(trusted_sources).most_common(1)
        source_hint = dominant_source[0][0].split(".")[0] if dominant_source else ""
        entity_hint = entity_hints[0] if entity_hints else ""
        query_candidates = []
        for keyword in keywords[:4]:
            query_candidates.append(" ".join(dict.fromkeys(part.strip() for part in [parent_query, keyword] if part.strip())))
            if source_hint and len(keyword.split()) >= 2:
                query_candidates.append(
                    " ".join(dict.fromkeys(part.strip() for part in [parent_query, keyword, source_hint] if part.strip()))
                )
            if entity_hint and entity_hint not in keyword:
                query_candidates.append(
                    " ".join(dict.fromkeys(part.strip() for part in [parent_query, entity_hint, keyword] if part.strip()))
                )
        query_candidates.extend(
            " ".join(dict.fromkeys(part.strip() for part in [parent_query, *combo] if part.strip()))
            for combo in itertools.combinations(keywords[:4], 2)
        )
        query_candidates = list(dict.fromkeys(query_candidates))
        if self._reranker is None or not self._reranker_config.enabled:
            return query_candidates[0]
        return self._reranker.rank(parent_query=parent_query, docs=docs, candidates=query_candidates)[0]
