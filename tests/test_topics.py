from __future__ import annotations

import numpy as np
import torch

from jakal_search.config import ThemeTokenConfig, TopicRerankerConfig
from jakal_search.theme_tokens import ThemeTokenInducer
from jakal_search.topics import KeywordTopicBuilder
from jakal_search.types import SearchDocument
from jakal_search.utils import normalize_rows


class FakeTransformerEmbedder:
    using_transformer = True

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(6, dtype=np.float32)
            if "search system" in lowered:
                vector[0] = 1.0
            if "policy" in lowered or "governance" in lowered:
                vector[0] += 0.8
                vector[1] = 1.0
            if "vector" in lowered or "tracking" in lowered:
                vector[2] = 1.0
            if "supporting explanation" in lowered or "detailed evidence" in lowered:
                vector[3] = 1.0
            rows.append(vector)
        return normalize_rows(np.asarray(rows, dtype=np.float32))


def make_doc(title: str, content: str) -> SearchDocument:
    return SearchDocument(
        title=title,
        snippet=title,
        content=content,
        url=f"https://example.com/{title.replace(' ', '-')}",
        source="example.com",
        query="search system",
        rank=1,
    )


def test_topic_builder_uses_transformer_alignment_for_keywords() -> None:
    docs = [
        make_doc(
            "policy governance roadmap",
            "policy governance roadmap with detailed evidence and supporting explanation for search system operators",
        ),
        make_doc(
            "policy standards",
            "policy standards and governance checkpoints for search system reliability",
        ),
        make_doc(
            "tracking vector drift",
            "tracking vector drift for search system branches",
        ),
    ]
    embedder = FakeTransformerEmbedder()
    for doc, vector in zip(docs, embedder.embed([doc.text for doc in docs])):
        doc.embedding = vector
        doc.trust_score = 0.8

    builder = KeywordTopicBuilder(embedder, TopicRerankerConfig(enabled=False))
    topic = builder.build("search system", docs[:2])

    assert topic.keywords
    assert topic.candidate_scores
    assert all("candidate" in item and "score" in item for item in topic.candidate_scores)
    assert any("policy" in keyword or "governance" in keyword for keyword in topic.keywords)
    assert "policy" in topic.query or "governance" in topic.query


def test_topic_builder_decodes_topics_from_theme_graph_vectors() -> None:
    docs = [
        make_doc(
            "policy governance roadmap",
            "policy governance roadmap with detailed evidence and supporting explanation for search system operators",
        ),
        make_doc(
            "policy standards",
            "policy standards and governance checkpoints for search system reliability",
        ),
        make_doc(
            "tracking vector drift",
            "tracking vector drift for search system branches",
        ),
    ]
    embedder = FakeTransformerEmbedder()
    for doc, vector in zip(docs, embedder.embed([doc.text for doc in docs])):
        doc.embedding = vector
        doc.trust_score = 0.8
        doc.retrieval_score = 0.7
        doc.freshness_score = 0.6

    theme_tokens = ThemeTokenInducer(ThemeTokenConfig()).induce(docs[:2])
    builder = KeywordTopicBuilder(embedder, TopicRerankerConfig(enabled=False))
    proposals = builder.decode_theme_tokens("search system", docs[:2], theme_tokens)

    assert proposals
    assert all(token.query for token in theme_tokens)
    assert all(token.candidate_scores for token in theme_tokens)
    assert any("policy" in token.label or "governance" in token.label for token in theme_tokens)


def test_theme_state_space_supports_gradients_through_topic_latents() -> None:
    docs = [
        make_doc(
            "policy governance roadmap",
            "policy governance roadmap with detailed evidence and supporting explanation for search system operators",
        ),
        make_doc(
            "tracking vector drift",
            "tracking vector drift for search system branches and monitoring",
        ),
    ]
    embedder = FakeTransformerEmbedder()
    for doc, vector in zip(docs, embedder.embed([doc.text for doc in docs])):
        doc.embedding = vector
        doc.trust_score = 0.8
        doc.retrieval_score = 0.7
        doc.freshness_score = 0.6

    inducer = ThemeTokenInducer(ThemeTokenConfig())
    asset_vector = np.asarray(embedder.embed(["search system"])[0], dtype=np.float32)
    asset_tensor = torch.tensor(asset_vector, dtype=torch.float32, requires_grad=True)
    output = inducer.forward_documents(docs, asset_vector=asset_tensor, embedder=embedder)
    assert output is not None
    loss = output.topic_latents.sum() + output.topic_scores.sum()
    loss.backward()

    assert inducer.state_space is not None
    assert asset_tensor.grad is not None
    assert float(asset_tensor.grad.abs().sum()) > 0.0
    assert inducer.state_space.theme_vectors.grad is not None
    assert float(inducer.state_space.theme_vectors.grad.abs().sum()) > 0.0
    gradients = [parameter.grad for parameter in inducer.state_space.topic_decoder.parameters()]
    assert gradients
    assert any(gradient is not None and float(gradient.abs().sum()) > 0.0 for gradient in gradients)
