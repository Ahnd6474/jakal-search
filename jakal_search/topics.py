from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .types import SearchDocument, TopicProposal


class KeywordTopicBuilder:
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
        label = ", ".join(keywords[:3]) if keywords else docs[0].title[:60]
        query_parts = [parent_query, *keywords[:3]]
        query = " ".join(dict.fromkeys(part.strip() for part in query_parts if part.strip()))
        return TopicProposal(
            label=label,
            query=query,
            keywords=keywords,
            evidence_document_ids=[doc.document_id for doc in docs[:3]],
        )

    def _topic_texts(self, docs: list[SearchDocument]) -> list[str]:
        texts: list[str] = []
        for doc in docs:
            if doc.title:
                # Titles carry denser topic intent, so we up-weight them.
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
