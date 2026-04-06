from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jakal_search.config import TrustConfig
from jakal_search.trust import EmbeddingTrustMLP, SourceTrustScorer
from jakal_search.trust_training import (
    TrustDatasetExample,
    balance_examples,
    build_training_matrix,
    dedupe_examples,
    evaluate_model,
    extract_email_text,
)


class FakeTransformerEmbedder:
    using_transformer = True

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for index, _ in enumerate(texts):
            rows.append(np.asarray([1.0 + index, 0.5], dtype=np.float32))
        return np.asarray(rows, dtype=np.float32)


def test_dedupe_examples_removes_short_and_duplicate_rows() -> None:
    examples = [
        TrustDatasetExample(text="hello world", label=1, source="a", source_url="u", item_id="1"),
        TrustDatasetExample(text="This is a long trusted text example", label=1, source="a", source_url="u", item_id="2"),
        TrustDatasetExample(text="This is a long trusted text example", label=1, source="b", source_url="v", item_id="3"),
        TrustDatasetExample(text="This is a long spam text example", label=0, source="c", source_url="w", item_id="4"),
    ]

    deduped = dedupe_examples(examples)

    assert len(deduped) == 2


def test_extract_email_text_combines_subject_and_body() -> None:
    payload = b"Subject: Trust alert\r\n\r\nThis is the body with useful details."

    text = extract_email_text(payload)

    assert "Trust alert" in text
    assert "useful details" in text


def test_source_trust_scorer_loads_supervised_head(tmp_path: Path) -> None:
    model = EmbeddingTrustMLP(2, 4)
    model_path = tmp_path / "trust_head.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 2,
            "hidden_dim": 4,
            "threshold": 0.61,
        },
        model_path,
    )

    scorer = SourceTrustScorer(TrustConfig(mlp_model_path=str(model_path)), FakeTransformerEmbedder())

    assert scorer._embedding_trust_head is not None
    assert scorer._loaded_model is True
    assert scorer._embedding_threshold == 0.61


def test_evaluate_model_returns_threshold_and_metrics() -> None:
    model = EmbeddingTrustMLP(2, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.asarray([1.0, 0.0], dtype=np.float32)

    threshold, metrics = evaluate_model(model, features, labels)

    assert 0.3 <= threshold <= 0.8
    assert metrics.val_size == 2


def test_balance_examples_caps_sources_and_labels() -> None:
    examples = []
    for index in range(5):
        examples.append(TrustDatasetExample(text=f"trusted example {index} long enough", label=1, source="docs", source_url="https://docs.python.org", item_id=str(index)))
        examples.append(TrustDatasetExample(text=f"spam example {index} long enough", label=0, source="spam", source_url="https://blogspot.com", item_id=f"s{index}"))

    balanced = balance_examples(examples, max_per_source=3, balance_labels=True, seed=0)

    assert len([item for item in balanced if item.source == "docs"]) <= 3
    assert len([item for item in balanced if item.source == "spam"]) <= 3
    assert sum(item.label == 1 for item in balanced) == sum(item.label == 0 for item in balanced)


def test_build_training_matrix_appends_site_features() -> None:
    examples = [
        TrustDatasetExample(
            text="trusted documentation example long enough",
            label=1,
            source="docs",
            source_url="https://docs.python.org/3/tutorial/index.html",
            item_id="1",
        )
    ]
    matrix = build_training_matrix(examples, np.asarray([[1.0, 0.5]], dtype=np.float32), config=TrustConfig())

    assert matrix.shape[0] == 1
    assert matrix.shape[1] > 2
