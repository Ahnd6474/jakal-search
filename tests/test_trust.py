from __future__ import annotations

import numpy as np

from jakal_search.config import TrustConfig
from jakal_search.trust import SourceTrustScorer
from jakal_search.types import SearchDocument
from jakal_search.utils import normalize_rows


def make_doc(title: str, source: str) -> SearchDocument:
    return SearchDocument(
        title=title,
        snippet=title,
        url=f"https://{source}/{title.replace(' ', '-')}",
        source=source,
        query="trust test",
        rank=1,
    )


class HeuristicEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(4, dtype=np.float32)
            if "secret" in lowered or "conspiracy" in lowered or "spam" in lowered:
                vector[0] = 1.0
            else:
                vector[1] = 1.0
            rows.append(vector)
        return normalize_rows(np.asarray(rows, dtype=np.float32))


class FakeTransformerEmbedder:
    using_transformer = True

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(6, dtype=np.float32)
            if any(token in lowered for token in ("peer reviewed", "official", "documentation", "government", "academic")):
                vector[0] = 1.0
                vector[1] = 0.8
            elif any(token in lowered for token in ("secret", "conspiracy", "spam", "giveaway", "urgent", "hack")):
                vector[2] = 1.0
                vector[3] = 0.8
            elif source_is_trusted_text(lowered):
                vector[0] = 0.9
                vector[4] = 0.4
            else:
                vector[2] = 0.9
                vector[5] = 0.4
            rows.append(vector)
        return normalize_rows(np.asarray(rows, dtype=np.float32))


def source_is_trusted_text(text: str) -> bool:
    return any(token in text for token in ("research", "documentation", "evidence", "maintainer", "policy"))


def test_trust_scorer_keeps_heuristic_path_for_non_transformer_embedder() -> None:
    scorer = SourceTrustScorer(TrustConfig(), HeuristicEmbedder())
    docs = [
        make_doc("research documentation evidence", "acm.org"),
        make_doc("secret spam conspiracy", "blogspot.com"),
    ]
    for doc, embedding in zip(docs, HeuristicEmbedder().embed([doc.text for doc in docs])):
        doc.embedding = embedding

    kept_docs = scorer.assess_documents(docs)

    assert len(kept_docs) == 1
    assert kept_docs[0].source == "acm.org"


def test_trust_scorer_uses_embedding_mlp_for_transformer_embeddings() -> None:
    config = TrustConfig(mlp_epochs=40, mlp_hidden_dim=16)
    scorer = SourceTrustScorer(config, FakeTransformerEmbedder())
    docs = [
        make_doc("research documentation evidence", "acm.org"),
        make_doc("secret spam conspiracy", "blogspot.com"),
    ]
    for doc, embedding in zip(docs, FakeTransformerEmbedder().embed([doc.text for doc in docs])):
        doc.embedding = embedding

    kept_docs = scorer.assess_documents(docs)

    assert scorer._embedding_trust_head is not None
    assert len(kept_docs) == 1
    assert kept_docs[0].source == "acm.org"
    assert docs[0].trust_score > docs[1].trust_score


def test_trust_scorer_uses_source_features_with_same_text() -> None:
    config = TrustConfig(mlp_epochs=40, mlp_hidden_dim=16)
    scorer = SourceTrustScorer(config, FakeTransformerEmbedder())
    docs = [
        make_doc("official documentation evidence", "docs.python.org"),
        make_doc("official documentation evidence", "blogspot.com"),
    ]
    for doc, embedding in zip(docs, FakeTransformerEmbedder().embed([doc.text for doc in docs])):
        doc.embedding = embedding

    kept_docs = scorer.assess_documents(docs)

    assert any(doc.source == "docs.python.org" for doc in kept_docs)
    assert docs[0].trust_score > docs[1].trust_score
