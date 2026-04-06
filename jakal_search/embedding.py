from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .utils import normalize_rows


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class HashingEmbedder:
    def __init__(self, n_features: int = 384) -> None:
        self._vectorizer = HashingVectorizer(
            alternate_sign=False,
            n_features=n_features,
            norm=None,
            stop_words="english",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        matrix = self._vectorizer.transform(texts)
        return normalize_rows(matrix.toarray().astype(np.float32))


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        fallback: TextEmbedder | None = None,
    ) -> None:
        self._fallback = fallback or HashingEmbedder()
        self._model = None
        self._device = resolve_transformer_device(device)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name, device=self._device)
        except ImportError:
            self._model = None

    @property
    def using_transformer(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | Any:
        return self._device

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self._model is None:
            return self._fallback.embed(texts)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return normalize_rows(np.asarray(vectors, dtype=np.float32))


def resolve_transformer_device(device: str | None) -> str | Any:
    requested = "auto" if device is None else device.strip().lower()
    if requested == "auto":
        detected = _auto_detect_transformer_device()
        return "cpu" if detected is None else detected
    if requested == "directml":
        detected = _get_directml_device()
        return "cpu" if detected is None else detected
    return requested


def _auto_detect_transformer_device() -> str | Any | None:
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    directml_device = _get_directml_device()
    if directml_device is not None:
        return directml_device
    return "cpu"


def _get_directml_device() -> Any | None:
    try:
        import torch_directml
    except ImportError:
        return None

    try:
        return torch_directml.device()
    except Exception:
        return None
