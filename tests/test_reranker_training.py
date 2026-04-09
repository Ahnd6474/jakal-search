from __future__ import annotations

from pathlib import Path

from jakal_search.config import TopicRerankerConfig
from jakal_search.reranker_training import build_reranker_examples, train_reranker
from jakal_search.reranking import TopicQueryReranker
from jakal_search.tuning import KeywordBenchmarkEmbedder, SEARCH_SYSTEM_DOCS


def test_build_reranker_examples_contains_labels() -> None:
    rows = build_reranker_examples(
        embedder_kind="synthetic",
        model_name="unused",
        device="cpu",
        seed=0,
    )

    labels = {row[1] for row in rows}
    assert labels == {0, 1}
    assert len(rows) >= 20


def test_topic_query_reranker_prefers_relevant_candidate(local_tmp_path: Path) -> None:
    rows = build_reranker_examples(
        embedder_kind="synthetic",
        model_name="unused",
        device="cpu",
        seed=0,
    )
    model_path, metrics = train_reranker(
        rows,
        local_tmp_path,
        hidden_dim=12,
        epochs=20,
        learning_rate=1e-2,
        weight_decay=1e-4,
        val_ratio=0.25,
        seed=0,
    )
    docs = [doc.clone() for doc in SEARCH_SYSTEM_DOCS[:4]]
    embedder = KeywordBenchmarkEmbedder()
    vectors = embedder.embed([doc.text for doc in docs])
    for doc, vector in zip(docs, vectors):
        doc.embedding = vector
        doc.trust_score = 0.8

    reranker = TopicQueryReranker(TopicRerankerConfig(model_path=str(model_path)), embedder)
    ranked = reranker.rank(
        parent_query="search system",
        docs=docs,
        candidates=["miracle hack", "policy governance", "search system guide"],
    )

    assert metrics.val_size > 0
    assert ranked[0] == "policy governance"
