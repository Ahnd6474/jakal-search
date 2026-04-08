from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from jakal_search.config import FalsehoodConfig
from jakal_search.falsehood import ClaimFalsehoodMLP, ClaimFalsehoodScorer
from jakal_search.falsehood_training import (
    ClaimDatasetExample,
    balance_examples,
    dedupe_examples,
    evaluate_model,
)
from jakal_search.types import SearchDocument


def test_claim_dataset_helpers_balance_and_dedupe() -> None:
    examples = [
        ClaimDatasetExample("false claim example long enough", 1, "a", "u", "1"),
        ClaimDatasetExample("false claim example long enough", 1, "a", "u", "2"),
        ClaimDatasetExample("true claim example long enough", 0, "b", "v", "3"),
        ClaimDatasetExample("another true claim example", 0, "b", "v", "4"),
    ]

    deduped = dedupe_examples(examples)
    balanced = balance_examples(deduped, max_per_source=2, seed=0)

    assert len(deduped) == 3
    assert sum(item.label == 1 for item in balanced) == sum(item.label == 0 for item in balanced)


def test_claim_falsehood_scorer_loads_head_and_scores(tmp_path: Path) -> None:
    model = ClaimFalsehoodMLP(2, 4)
    model_path = tmp_path / "claim_falsehood.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 2,
            "hidden_dim": 4,
            "threshold": 0.61,
        },
        model_path,
    )
    scorer = ClaimFalsehoodScorer(FalsehoodConfig(model_path=str(model_path), block_high_risk=False))
    doc = SearchDocument(
        title="claim",
        snippet="claim",
        url="https://example.com/claim",
        source="example.com",
        query="q",
        rank=1,
    )
    doc.embedding = np.asarray([1.0, 0.5], dtype=np.float32)

    scored = scorer.assess_documents([doc])

    assert scorer.enabled is True
    assert scored[0].claim_falsehood_score >= 0.0
    assert scored[0].trust_score <= 0.5


def test_falsehood_evaluate_model_returns_metrics() -> None:
    model = ClaimFalsehoodMLP(2, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    threshold, metrics = evaluate_model(
        model,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert 0.3 <= threshold <= 0.8
    assert metrics.val_size == 2
