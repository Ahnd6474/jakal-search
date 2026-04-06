from __future__ import annotations

import numpy as np

from jakal_search.tuning import CachingEmbedder, build_embedder, default_tuning_config, evaluate_config, tune


def test_evaluate_config_scores_all_benchmark_cases() -> None:
    result = evaluate_config(default_tuning_config())

    assert result.score > 0
    assert len(result.evaluations) == 4
    assert len(result.trust_evaluations) == 2
    assert all(item.score > 0 for item in result.evaluations)
    assert all(item.score > 0 for item in result.trust_evaluations)


def test_tune_returns_ranked_results() -> None:
    best, top_results = tune(limit=8)

    assert best.score == top_results[0].score
    assert top_results
    assert best.config.limits.results_per_query in {8, 10}


def test_caching_embedder_reuses_previous_embeddings() -> None:
    calls: list[list[str]] = []

    class FakeEmbedder:
        def embed(self, texts: list[str]) -> np.ndarray:
            calls.append(list(texts))
            return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    embedder = CachingEmbedder(FakeEmbedder())
    first = embedder.embed(["alpha", "beta"])
    second = embedder.embed(["beta", "gamma"])

    assert first.shape == (2, 2)
    assert second.shape == (2, 2)
    assert calls == [["alpha", "beta"], ["gamma"]]


def test_build_embedder_can_wrap_sentence_transformer(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeSentenceTransformerEmbedder:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr("jakal_search.tuning.SentenceTransformerEmbedder", FakeSentenceTransformerEmbedder)

    embedder = build_embedder("sentence-transformer", "model-z", "directml")

    assert captured == {"model_name": "model-z", "device": "directml"}
    assert embedder.embed(["x"]).shape == (1, 1)
