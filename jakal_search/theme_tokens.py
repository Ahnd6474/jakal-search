from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ThemeTokenConfig
from .retrieval import tokenize_text
from .types import SearchDocument, ThemeToken
from .utils import normalize_rows, normalize_vector, stable_hash

_DEFAULT_THEMES: tuple[tuple[str, str], ...] = (
    ("policy", "policy governance regulation standards compliance oversight"),
    ("tracking", "tracking metrics monitoring observability telemetry drift vectors"),
    ("market", "demand adoption market revenue earnings growth customers"),
    ("operations", "deployment rollout container infrastructure operations platform"),
    ("community", "community maintainer contributors moderation support collaboration"),
    ("risk", "risk safety failure incident bug security reliability"),
    ("research", "research benchmark evaluation paper methodology experiments evidence"),
    ("product", "product roadmap features capability architecture engineering"),
)


@dataclass(slots=True)
class GlobalTheme:
    theme_id: str
    descriptor: str
    vector: np.ndarray


@dataclass(slots=True)
class ThemeForwardOutput:
    theme_ids: list[str]
    theme_vectors: torch.Tensor
    edge_matrix: torch.Tensor
    memberships: torch.Tensor
    attention_weights: torch.Tensor
    pooled_docs: torch.Tensor
    base_state: torch.Tensor
    state: torch.Tensor
    topic_latents: torch.Tensor
    topic_scores: torch.Tensor


def _descriptor_vector(descriptor: str, embedding_dim: int) -> np.ndarray:
    vector = np.zeros((embedding_dim,), dtype=np.float32)
    tokens = tokenize_text(descriptor)
    if not tokens:
        return vector
    for index, token in enumerate(tokens, start=1):
        digest = stable_hash(f"{token}:{index}")
        bucket = int(digest[:8], 16) % embedding_dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + (0.1 * min(len(token), 8)))
    return normalize_vector(vector)


class GlobalThemeBank:
    def __init__(self, config: ThemeTokenConfig) -> None:
        self._config = config
        self._themes: list[GlobalTheme] = []
        self._embedding_dim = 0

    def ensure(self, *, embedding_dim: int, embedder=None) -> list[GlobalTheme]:
        if self._themes and self._embedding_dim == embedding_dim:
            return self._themes
        descriptors = [descriptor for _, descriptor in _DEFAULT_THEMES]
        if embedder is not None:
            try:
                vectors = normalize_rows(np.asarray(embedder.embed(descriptors), dtype=np.float32))
            except Exception:
                vectors = self._fallback_vectors(descriptors, embedding_dim)
        else:
            vectors = self._fallback_vectors(descriptors, embedding_dim)
        if vectors.shape[1] != embedding_dim:
            vectors = self._fallback_vectors(descriptors, embedding_dim)
        self._themes = [
            GlobalTheme(theme_id=theme_id, descriptor=descriptor, vector=vectors[index].copy())
            for index, (theme_id, descriptor) in enumerate(_DEFAULT_THEMES)
        ]
        self._embedding_dim = embedding_dim
        return self._themes

    def _fallback_vectors(self, descriptors: list[str], embedding_dim: int) -> np.ndarray:
        rows = np.asarray([_descriptor_vector(descriptor, embedding_dim) for descriptor in descriptors], dtype=np.float32)
        return normalize_rows(rows)


class ThemeStateSpace(nn.Module):
    def __init__(self, theme_vectors: np.ndarray, config: ThemeTokenConfig) -> None:
        super().__init__()
        normalized = normalize_rows(np.asarray(theme_vectors, dtype=np.float32))
        self.config = config
        self.embedding_dim = normalized.shape[1]
        self.theme_count = normalized.shape[0]
        self.hidden_dim = max(6, min(32, self.embedding_dim))
        self.theme_vectors = nn.Parameter(torch.from_numpy(normalized))
        self.register_buffer(
            "source_projection",
            torch.roll(torch.eye(self.embedding_dim, dtype=torch.float32), shifts=1, dims=1),
        )
        self.register_buffer(
            "target_projection",
            torch.roll(torch.eye(self.embedding_dim, dtype=torch.float32), shifts=2, dims=1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.topic_decoder = nn.Sequential(
            nn.Linear(self.embedding_dim * 4 + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        self.topic_score_head = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        doc_vectors: torch.Tensor,
        doc_weights: torch.Tensor,
        *,
        asset_vector: torch.Tensor | None = None,
    ) -> ThemeForwardOutput:
        theme_vectors = F.normalize(self.theme_vectors, dim=1)
        keyed_docs = self.asset_conditioned_keys(doc_vectors, asset_vector)
        attention_logits = torch.matmul(keyed_docs, theme_vectors.T) / max(self.config.temperature, 1e-3)
        weighted_logits = attention_logits + torch.log(torch.clamp(doc_weights[:, None], min=1e-8))
        attention_weights = torch.softmax(weighted_logits, dim=0)
        memberships = torch.softmax(attention_logits, dim=1)
        pooled_docs = torch.matmul(attention_weights.T, doc_vectors)
        pooled_docs = F.normalize(pooled_docs, dim=1)
        base_alignment = torch.relu(torch.sum(theme_vectors * pooled_docs, dim=1))
        state_mass = torch.sum(attention_weights * doc_weights[:, None], dim=0)
        base_state = base_alignment * state_mass
        edge_matrix = self.edge_matrix(theme_vectors)
        state = self.propagate_state(base_state, edge_matrix)
        topic_latents = self.decode_topics(theme_vectors, pooled_docs, state, asset_vector)
        topic_scores = self.score_topics(topic_latents, pooled_docs, state)
        return ThemeForwardOutput(
            theme_ids=[theme_id for theme_id, _ in _DEFAULT_THEMES],
            theme_vectors=theme_vectors,
            edge_matrix=edge_matrix,
            memberships=memberships,
            attention_weights=attention_weights,
            pooled_docs=pooled_docs,
            base_state=base_state,
            state=state,
            topic_latents=topic_latents,
            topic_scores=topic_scores,
        )

    def asset_conditioned_keys(
        self,
        doc_vectors: torch.Tensor,
        asset_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        if asset_vector is None or asset_vector.numel() == 0 or asset_vector.shape[0] != doc_vectors.shape[1]:
            return F.normalize(doc_vectors, dim=1)
        asset = F.normalize(asset_vector, dim=0)
        projection_strength = torch.matmul(doc_vectors, asset.unsqueeze(1))
        projected = projection_strength * asset.unsqueeze(0)
        return F.normalize(doc_vectors + projected, dim=1)

    def edge_matrix(self, theme_vectors: torch.Tensor) -> torch.Tensor:
        source_vectors = F.normalize(torch.matmul(theme_vectors, self.source_projection), dim=1)
        target_vectors = F.normalize(torch.matmul(theme_vectors, self.target_projection), dim=1)
        edges = torch.zeros((self.theme_count, self.theme_count), dtype=theme_vectors.dtype, device=theme_vectors.device)
        for source_index in range(self.theme_count):
            for target_index in range(self.theme_count):
                if source_index == target_index:
                    continue
                default_score = torch.sum(source_vectors[source_index] * target_vectors[target_index])
                features = torch.cat(
                    [
                        theme_vectors[source_index],
                        theme_vectors[target_index],
                        theme_vectors[source_index] - theme_vectors[target_index],
                        theme_vectors[source_index] * theme_vectors[target_index],
                    ],
                    dim=0,
                )
                correction = self.edge_mlp(features).squeeze(-1)
                edges[source_index, target_index] = F.softsign(default_score + (0.2 * correction))
            row_norm = torch.sum(torch.abs(edges[source_index]))
            if float(row_norm.item()) > 1.0:
                edges[source_index] = edges[source_index] / row_norm
        return edges

    def propagate_state(self, base_state: torch.Tensor, edge_matrix: torch.Tensor) -> torch.Tensor:
        state = base_state
        for _ in range(max(self.config.graph_steps, 0)):
            messages = torch.matmul(edge_matrix.T, state)
            state = torch.relu(state + messages)
        return state

    def decode_topics(
        self,
        theme_vectors: torch.Tensor,
        pooled_docs: torch.Tensor,
        state: torch.Tensor,
        asset_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        if asset_vector is None or asset_vector.numel() == 0 or asset_vector.shape[0] != theme_vectors.shape[1]:
            asset = torch.zeros_like(theme_vectors)
        else:
            asset = F.normalize(asset_vector, dim=0).unsqueeze(0).expand_as(theme_vectors)
        state_column = state.unsqueeze(1)
        decoder_input = torch.cat(
            [
                theme_vectors,
                pooled_docs,
                asset,
                theme_vectors * pooled_docs,
                state_column,
            ],
            dim=1,
        )
        topic_latents = self.topic_decoder(decoder_input)
        return F.normalize(topic_latents + pooled_docs + theme_vectors, dim=1)

    def score_topics(
        self,
        topic_latents: torch.Tensor,
        pooled_docs: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        score_input = torch.cat([topic_latents, pooled_docs, state.unsqueeze(1)], dim=1)
        return self.topic_score_head(score_input).squeeze(-1)


class ThemeTokenInducer:
    def __init__(self, config: ThemeTokenConfig) -> None:
        self._config = config
        self._theme_bank = GlobalThemeBank(config)
        self._state_space: ThemeStateSpace | None = None
        self.last_state: np.ndarray = np.empty((0,), dtype=np.float32)
        self.last_state_tensor: torch.Tensor | None = None

    @property
    def state_space(self) -> ThemeStateSpace | None:
        return self._state_space

    def forward_documents(
        self,
        docs: list[SearchDocument],
        *,
        asset_vector: np.ndarray | torch.Tensor | None = None,
        embedder=None,
    ) -> ThemeForwardOutput | None:
        if not self._config.enabled or not docs:
            self.last_state = np.empty((0,), dtype=np.float32)
            self.last_state_tensor = None
            return None
        doc_matrix = np.asarray([doc.embedding for doc in docs if doc.embedding is not None], dtype=np.float32)
        if doc_matrix.size == 0:
            self.last_state = np.empty((0,), dtype=np.float32)
            self.last_state_tensor = None
            return None
        doc_matrix = normalize_rows(doc_matrix)
        themes = self._theme_bank.ensure(embedding_dim=doc_matrix.shape[1], embedder=embedder)
        if self._state_space is None or self._state_space.embedding_dim != doc_matrix.shape[1]:
            self._state_space = ThemeStateSpace(
                np.asarray([theme.vector for theme in themes], dtype=np.float32),
                self._config,
            )
        doc_vectors = torch.as_tensor(doc_matrix, dtype=torch.float32)
        doc_weights = torch.as_tensor(self._document_weights(docs), dtype=torch.float32)
        asset_tensor: torch.Tensor | None = None
        if isinstance(asset_vector, torch.Tensor):
            if asset_vector.numel() > 0 and asset_vector.shape[0] == doc_matrix.shape[1]:
                asset_tensor = F.normalize(asset_vector.to(dtype=torch.float32), dim=0)
        elif asset_vector is not None and asset_vector.size > 0 and asset_vector.shape[0] == doc_matrix.shape[1]:
            asset_tensor = torch.as_tensor(normalize_vector(asset_vector.astype(np.float32)), dtype=torch.float32)
        output = self._state_space(doc_vectors, doc_weights, asset_vector=asset_tensor)
        self.last_state_tensor = output.state
        self.last_state = output.state.detach().cpu().numpy().astype(np.float32, copy=True)
        return output

    def induce(
        self,
        docs: list[SearchDocument],
        *,
        asset_vector: np.ndarray | None = None,
        asset_label: str = "",
        embedder=None,
    ) -> list[ThemeToken]:
        output = self.forward_documents(docs, asset_vector=asset_vector, embedder=embedder)
        if output is None:
            return []

        theme_ids = output.theme_ids
        attention_weights = output.attention_weights.detach().cpu().numpy()
        memberships = output.memberships.detach().cpu().numpy()
        pooled_docs = output.pooled_docs.detach().cpu().numpy()
        edge_matrix = output.edge_matrix.detach().cpu().numpy()
        base_state = output.base_state.detach().cpu().numpy()
        state = output.state.detach().cpu().numpy()
        theme_vectors = output.theme_vectors.detach().cpu().numpy()
        topic_latents = output.topic_latents.detach().cpu().numpy()
        topic_scores = output.topic_scores.detach().cpu().numpy()

        tokens: list[ThemeToken] = []
        for theme_index, theme_id in enumerate(theme_ids):
            attention_column = attention_weights[:, theme_index]
            membership_column = memberships[:, theme_index]
            ranked_indexes = sorted(range(len(docs)), key=lambda index: float(attention_column[index]), reverse=True)
            anchor_indexes = ranked_indexes[:3]
            member_indexes = [index for index in ranked_indexes if membership_column[index] >= self._config.min_membership] or anchor_indexes[:1]
            context_vector = pooled_docs[theme_index]
            keywords = self._keywords_for_members(docs, membership_column, member_indexes)
            token = ThemeToken(
                token_id=theme_id,
                label=theme_id,
                query=theme_id,
                score=float(state[theme_index]),
                base_score=float(base_state[theme_index]),
                asset_condition=asset_label,
                state_norm=float(state[theme_index]),
                topic_norm=float(np.linalg.norm(topic_latents[theme_index])),
                anchor_document_ids=[docs[index].document_id for index in anchor_indexes],
                anchor_titles=[docs[index].title for index in anchor_indexes],
                attention_document_ids=[docs[index].document_id for index in anchor_indexes],
                attention_titles=[docs[index].title for index in anchor_indexes],
                member_count=len(member_indexes),
                top_keywords=keywords,
                candidate_scores=[{"candidate": theme_id, "score": float(topic_scores[theme_index])}],
                outgoing_edges=self._edge_records(edge_matrix, theme_ids, theme_index),
                state_vector=theme_vectors[theme_index].copy(),
                topic_vector=topic_latents[theme_index].copy(),
            )
            tokens.append(token)

        for doc_index, doc in enumerate(docs):
            doc.metadata["theme_memberships"] = [
                {
                    "token_id": token.token_id,
                    "label": token.label,
                    "score": round(float(memberships[doc_index, token_index]), 6),
                    "base_score": round(float(attention_weights[doc_index, token_index]), 6),
                    "top_keywords": list(token.top_keywords),
                }
                for token_index, token in enumerate(tokens)
            ]
        return tokens

    def state_delta(self, previous_state: np.ndarray | torch.Tensor | None) -> float:
        if previous_state is None or self.last_state_tensor is None:
            return 1.0
        if isinstance(previous_state, np.ndarray):
            previous = torch.as_tensor(previous_state, dtype=torch.float32)
        else:
            previous = previous_state.detach()
        current = self.last_state_tensor.detach()
        if previous.shape != current.shape:
            return 1.0
        return float(torch.linalg.norm(current - previous).item())

    def _document_weights(self, docs: list[SearchDocument]) -> np.ndarray:
        weights = np.asarray(
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
        return weights / np.clip(weights.sum(), 1e-12, None)

    def _edge_records(
        self,
        edge_matrix: np.ndarray,
        theme_ids: list[str],
        source_index: int,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        if edge_matrix.size == 0:
            return records
        for target_index, theme_id in enumerate(theme_ids):
            if source_index == target_index:
                continue
            weight = float(edge_matrix[source_index, target_index])
            records.append({"target_token_id": theme_id, "weight": round(weight, 6)})
        records.sort(key=lambda item: abs(float(item["weight"])), reverse=True)
        return records

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
