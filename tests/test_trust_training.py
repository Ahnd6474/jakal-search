from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jakal_search.config import TrustConfig
from jakal_search.trust import EmbeddingTrustMLP, SourceTrustScorer
from jakal_search.trust_training import (
    TrustDatasetExample,
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
