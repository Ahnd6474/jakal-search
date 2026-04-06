from __future__ import annotations

import itertools

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TopicRerankerConfig
from .embedding import TextEmbedder
from .reranking import TopicQueryReranker
from .types import SearchDocument, TopicProposal


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
        texts = self._topic_texts(docs)
        if not texts:
            return TopicProposal(
                label=parent_query,
                query=parent_query,
                keywords=[],
                evidence_document_ids=[doc.document_id for doc in docs[:3]],
            )

        keywords = self._extract_keywords(texts, parent_query)
        reranked_keywords = self._rerank_keywords(parent_query, docs, keywords)
        label_terms = reranked_keywords[:3]
        label = ", ".join(label_terms) if label_terms else docs[0].title[:60]
        query = self._build_query(parent_query, docs, reranked_keywords)
        return TopicProposal(
            label=label,
            query=query,
            keywords=reranked_keywords,
            evidence_document_ids=[doc.document_id for doc in docs[:3]],
        )

    def _topic_texts(self, docs: list[SearchDocument]) -> list[str]:
        texts: list[str] = []
        for doc in docs:
            if doc.title:
                texts.extend([doc.title, doc.title])
            if doc.content:
                texts.append(doc.content)
        return texts

    def _extract_keywords(self, texts: list[str], parent_query: str) -> list[str]:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=48,
            )
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        scores = np.asarray(matrix.mean(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        parent_tokens = set(parent_query.lower().split())
        ranked = [
            feature
            for feature in features[np.argsort(scores)[::-1]]
            if not set(feature.lower().split()).issubset(parent_tokens)
        ]
        return ranked[:6]

    def _rerank_keywords(self, parent_query: str, docs: list[SearchDocument], keywords: list[str]) -> list[str]:
        if not keywords:
            return []
        if self._reranker is None or not self._reranker_config.enabled:
            return keywords
        candidates = keywords[: self._reranker_config.candidate_pool_size]
        return self._reranker.rank(parent_query=parent_query, docs=docs, candidates=candidates)

    def _build_query(self, parent_query: str, docs: list[SearchDocument], keywords: list[str]) -> str:
        if not keywords:
            return parent_query
        query_candidates = [
            " ".join(dict.fromkeys(part.strip() for part in [parent_query, keyword] if part.strip()))
            for keyword in keywords[:3]
        ]
        query_candidates.extend(
            " ".join(dict.fromkeys(part.strip() for part in [parent_query, *combo] if part.strip()))
            for combo in itertools.combinations(keywords[:4], 2)
        )
        query_candidates = list(dict.fromkeys(query_candidates))
        if self._reranker is None or not self._reranker_config.enabled:
            return query_candidates[0]
        return self._reranker.rank(parent_query=parent_query, docs=docs, candidates=query_candidates)[0]
