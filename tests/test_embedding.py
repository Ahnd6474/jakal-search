from __future__ import annotations

import types

from jakal_search.embedding import SentenceTransformerEmbedder, resolve_transformer_device
from jakal_search.engine import build_default_engine
from jakal_search.config import EngineConfig


def test_resolve_transformer_device_falls_back_to_cpu_when_directml_is_missing(monkeypatch) -> None:
    monkeypatch.setattr("jakal_search.embedding._get_directml_device", lambda: None)

    assert resolve_transformer_device("directml") == "cpu"


def test_sentence_transformer_embedder_passes_directml_device(monkeypatch) -> None:
    sentinel_device = object()
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: object) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

    monkeypatch.setattr("jakal_search.embedding._get_directml_device", lambda: sentinel_device)
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = SentenceTransformerEmbedder("test-model", device="directml")

    assert embedder.using_transformer is True
    assert embedder.device is sentinel_device
    assert captured == {"model_name": "test-model", "device": sentinel_device}


def test_build_default_engine_uses_configured_transformer_device(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeEmbedder:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

    monkeypatch.setattr("jakal_search.engine.SentenceTransformerEmbedder", FakeEmbedder)

    config = EngineConfig(transformer_model="model-a", transformer_device="directml")
    engine = build_default_engine(config)

    assert captured == {"model_name": "model-a", "device": "directml"}
    assert engine.config.transformer_device == "directml"


def test_sentence_transformer_embedder_uses_undecorated_encode_for_directml(monkeypatch) -> None:
    calls: list[str] = []

    class FakeModel:
        def __init__(self) -> None:
            def wrapped(_: object, texts: list[str], normalize_embeddings: bool) -> list[list[float]]:
                calls.append("wrapped")
                return [[1.0, 0.0] for _ in texts]

            def decorated(texts: list[str], normalize_embeddings: bool) -> list[list[float]]:
                calls.append("decorated")
                return [[0.0, 1.0] for _ in texts]

            decorated.__wrapped__ = wrapped  # type: ignore[attr-defined]
            self.encode = decorated

    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder._model = FakeModel()
    embedder._device = types.SimpleNamespace(type="privateuseone")

    vectors = embedder.embed(["a", "b"])

    assert calls == ["wrapped"]
    assert vectors.shape == (2, 2)
