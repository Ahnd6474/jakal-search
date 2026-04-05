from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .types import SearchDocument


class KeywordTopicBuilder:
    def build(self, parent_query: str, docs: list[SearchDocument]) -> tuple[str, str, list[str]]:
        texts = [doc.text for doc in docs if doc.text]
        if not texts:
            label = parent_query
            return label, parent_query, []

        keywords = self._extract_keywords(texts, parent_query)
        label = ", ".join(keywords[:3]) if keywords else docs[0].title[:60]
        query_parts = [parent_query, *keywords[:3]]
        query = " ".join(dict.fromkeys(part.strip() for part in query_parts if part.strip()))
        return label, query, keywords

    def _extract_keywords(self, texts: list[str], parent_query: str) -> list[str]:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=32,
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
