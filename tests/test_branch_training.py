from __future__ import annotations

from pathlib import Path

from jakal_search.branch_training import build_branch_examples, train_branch_head
from jakal_search.config import ScoringConfig
from jakal_search.scoring import BranchDecisionModel
from jakal_search.types import BranchMetrics


def test_build_branch_examples_has_positive_and_negative_rows() -> None:
    examples = build_branch_examples(
        embedder_kind="synthetic",
        model_name="unused",
        device="cpu",
        seed=0,
    )

    labels = {example.label for example in examples}
    assert labels == {0, 1}
    assert len(examples) >= 1000


def test_train_branch_head_and_load_runtime_model(tmp_path: Path) -> None:
    examples = build_branch_examples(
        embedder_kind="synthetic",
        model_name="unused",
        device="cpu",
        seed=0,
    )
    model_path, metrics = train_branch_head(
        examples,
        tmp_path,
        hidden_dim=12,
        epochs=20,
        learning_rate=1e-2,
        weight_decay=1e-4,
        val_ratio=0.25,
        seed=0,
    )

    model = BranchDecisionModel(ScoringConfig(mlp_model_path=str(model_path)))
    positive = BranchMetrics(0.9, 0.8, 0.8, 0.6, 0.8, 1.0, 0.1)
    negative = BranchMetrics(0.1, 0.2, 0.2, 0.2, 0.1, 0.3, 0.8)

    assert metrics.val_size > 0
    assert metrics.train_size > metrics.val_size
    assert model.evaluate(positive).probability > model.evaluate(negative).probability
