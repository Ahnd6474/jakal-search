from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .config import ThemeTokenConfig
from .retrieval import tokenize_text
from .types import SearchDocument, ThemeToken
from .utils import normalize_rows, normalize_vector


def _softmax_rows(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exp = np.exp(shifted)
    sums = np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
    return exp / sums


def _softmax_columns(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=0, keepdims=True)
    exp = np.exp(shifted)
    sums = np.clip(exp.sum(axis=0, keepdims=True), 1e-12, None)
    return exp / sums


class ThemeTokenInducer:
    def __init__(self, config: ThemeTokenConfig) -> None:
        self._config = config

    def induce(
        self,
        docs: list[SearchDocument],
        *,
        asset_vector: np.ndarray | None = None,
        asset_label: str = "",
    ) -> list[ThemeToken]:
        if not self._config.enabled or not docs:
            return []
        doc_vectors = np.asarray([doc.embedding for doc in docs if doc.embedding is not None], dtype=np.float32)
        if doc_vectors.size == 0:
            return []
        doc_vectors = normalize_rows(doc_vectors)
        token_count = self._token_count(len(docs))
        centroids = self._initialize_centroids(doc_vectors, docs, token_count)
        if centroids.size == 0:
            return []

        doc_weights = np.asarray(
            [
                max(
                    0.05,
                    (0.45 * float(doc.retrieval_score))
                    + (0.35 * float(doc.trust_score))
                    + (0.20 * float(doc.freshness_score)),
                )
                for doc in docs
            ],
            dtype=np.float32,
        )
        doc_weights = doc_weights / np.clip(doc_weights.sum(), 1e-12, None)
        memberships = np.zeros((len(docs), token_count), dtype=np.float32)

        for _ in range(max(self._config.iterations, 1)):
            similarities = np.matmul(doc_vectors, centroids.T) / max(self._config.temperature, 1e-3)
            memberships = _softmax_rows(similarities)
            weighted_memberships = memberships * doc_weights[:, None]
            new_centroids = np.matmul(weighted_memberships.T, doc_vectors)
            centroids = normalize_rows(new_centroids)

        initial_states = normalize_rows(np.matmul((memberships * doc_weights[:, None]).T, doc_vectors))
        edge_matrix = self._build_edge_matrix(initial_states, memberships)
        propagated_states = self._propagate(initial_states, edge_matrix)
        final_similarities = np.matmul(doc_vectors, propagated_states.T) / max(self._config.temperature, 1e-3)
        final_memberships = _softmax_rows(final_similarities)
        topic_vectors, attention_weights = self._pool_document_context(
            doc_vectors,
            doc_weights,
            propagated_states,
            asset_vector=asset_vector,
        )

        tokens: list[ThemeToken] = []
        for token_index in range(token_count):
            token_memberships = final_memberships[:, token_index]
            base_memberships = memberships[:, token_index]
            member_indexes = np.where(token_memberships >= self._config.min_membership)[0]
            if member_indexes.size == 0:
                member_indexes = np.asarray([int(np.argmax(token_memberships))], dtype=np.int64)
            sorted_indexes = sorted(member_indexes.tolist(), key=lambda index: float(token_memberships[index]), reverse=True)
            anchor_indexes = sorted_indexes[:3]
            attention_indexes = sorted(
                range(len(docs)),
                key=lambda index: float(attention_weights[index, token_index]),
                reverse=True,
            )[:3]
            score = float(np.mean(token_memberships[member_indexes]))
            keywords = self._keywords_for_members(docs, token_memberships, member_indexes.tolist())
            label = f"theme_{token_index:02d}"
            token = ThemeToken(
                token_id=label,
                label=label,
                score=score,
                base_score=float(np.mean(base_memberships)),
                asset_condition=asset_label,
                state_norm=float(np.linalg.norm(propagated_states[token_index])),
                topic_norm=float(np.linalg.norm(topic_vectors[token_index])),
                anchor_document_ids=[docs[index].document_id for index in anchor_indexes],
                anchor_titles=[docs[index].title for index in anchor_indexes],
                attention_document_ids=[docs[index].document_id for index in attention_indexes],
                attention_titles=[docs[index].title for index in attention_indexes],
                member_count=len(member_indexes),
                top_keywords=keywords,
                outgoing_edges=self._edge_records(edge_matrix, token_index),
                state_vector=propagated_states[token_index].copy(),
                topic_vector=topic_vectors[token_index].copy(),
            )
            tokens.append(token)

        for doc_index, doc in enumerate(docs):
            doc.metadata["theme_memberships"] = [
                {
                    "token_id": token.token_id,
                    "label": token.label,
                    "score": round(float(final_memberships[doc_index, token_index]), 6),
                    "base_score": round(float(memberships[doc_index, token_index]), 6),
                    "top_keywords": list(token.top_keywords),
                }
                for token_index, token in enumerate(tokens)
            ]
        return tokens

    def _pool_document_context(
        self,
        doc_vectors: np.ndarray,
        doc_weights: np.ndarray,
        propagated_states: np.ndarray,
        *,
        asset_vector: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        doc_keys = self._asset_conditioned_keys(doc_vectors, asset_vector)
        logits = np.matmul(doc_keys, propagated_states.T) / max(self._config.temperature, 1e-3)
        weighted_logits = logits + np.log(np.clip(doc_weights[:, None], 1e-8, None))
        attention_weights = _softmax_columns(weighted_logits)
        pooled_vectors = np.matmul(attention_weights.T, doc_vectors)
        topic_vectors = normalize_rows(propagated_states + pooled_vectors)
        return topic_vectors, attention_weights

    def _asset_conditioned_keys(
        self,
        doc_vectors: np.ndarray,
        asset_vector: np.ndarray | None,
    ) -> np.ndarray:
        if asset_vector is None or asset_vector.size == 0:
            return doc_vectors
        asset = normalize_vector(asset_vector.astype(np.float32))
        if asset.shape[0] != doc_vectors.shape[1]:
            return doc_vectors
        projection_strength = np.matmul(doc_vectors, asset[:, None])
        projected = projection_strength * asset[None, :]
        return normalize_rows(doc_vectors + projected)

    def _build_edge_matrix(self, token_states: np.ndarray, memberships: np.ndarray) -> np.ndarray:
        token_count = token_states.shape[0]
        if token_count == 0:
            return np.empty((0, 0), dtype=np.float32)
        base_scores = memberships.mean(axis=0)
        raw = np.zeros((token_count, token_count), dtype=np.float32)
        for source_index in range(token_count):
            for target_index in range(token_count):
                if source_index == target_index:
                    continue
                similarity = float(np.dot(token_states[source_index], token_states[target_index]))
                direction = float(base_scores[source_index] - base_scores[target_index])
                raw[source_index, target_index] = similarity + (self._config.edge_direction_weight * direction)
        for source_index in range(token_count):
            raw[source_index, source_index] = 0.0
            denom = float(np.sum(np.abs(raw[source_index])))
            if denom <= self._config.edge_epsilon:
                continue
            raw[source_index] = raw[source_index] / denom
            raw[source_index, source_index] = 0.0
        return raw

    def _propagate(self, token_states: np.ndarray, edge_matrix: np.ndarray) -> np.ndarray:
        propagated = token_states.astype(np.float32, copy=True)
        for _ in range(max(self._config.graph_steps, 0)):
            messages = np.matmul(edge_matrix.T, propagated)
            propagated = normalize_rows(propagated + messages)
        return propagated

    def _edge_records(self, edge_matrix: np.ndarray, source_index: int) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        if edge_matrix.size == 0:
            return records
        for target_index in range(edge_matrix.shape[1]):
            if source_index == target_index:
                continue
            weight = float(edge_matrix[source_index, target_index])
            records.append(
                {
                    "target_token_id": f"theme_{target_index:02d}",
                    "weight": round(weight, 6),
                }
            )
        records.sort(key=lambda item: abs(float(item["weight"])), reverse=True)
        return records

    def _token_count(self, doc_count: int) -> int:
        target = int(round(math.sqrt(max(doc_count, 1))))
        return max(self._config.min_tokens, min(self._config.max_tokens, target))

    def _initialize_centroids(
        self,
        doc_vectors: np.ndarray,
        docs: list[SearchDocument],
        token_count: int,
    ) -> np.ndarray:
        ranked_indexes = sorted(
            range(len(docs)),
            key=lambda index: (
                float(docs[index].retrieval_score),
                float(docs[index].trust_score),
                float(docs[index].freshness_score),
            ),
            reverse=True,
        )
        chosen: list[int] = []
        for index in ranked_indexes:
            if not chosen:
                chosen.append(index)
                continue
            max_similarity = max(float(np.dot(doc_vectors[index], doc_vectors[other])) for other in chosen)
            if max_similarity < 0.92 or len(chosen) < token_count:
                chosen.append(index)
            if len(chosen) >= token_count:
                break
        if not chosen:
            return np.empty((0, 0), dtype=np.float32)
        while len(chosen) < token_count:
            chosen.append(chosen[-1])
        return doc_vectors[np.asarray(chosen[:token_count], dtype=np.int64)]

    def _keywords_for_members(
        self,
        docs: list[SearchDocument],
        memberships: np.ndarray,
        member_indexes: list[int],
    ) -> list[str]:
        weighted_counts: Counter[str] = Counter()
        for index in member_indexes:
            weight = float(memberships[index])
            tokens = tokenize_text(docs[index].title + " " + (docs[index].snippet or ""))
            for token in tokens:
                if len(token) < 3:
                    continue
                weighted_counts[token] += weight
        return [token for token, _ in weighted_counts.most_common(4)]
