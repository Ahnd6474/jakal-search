from __future__ import annotations

from typing import Protocol

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
    def __init__(self, model_name: str, fallback: TextEmbedder | None = None) -> None:
        self._fallback = fallback or HashingEmbedder()
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
        except ImportError:
            self._model = None

    @property
    def using_transformer(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self._model is None:
            return self._fallback.embed(texts)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return normalize_rows(np.asarray(vectors, dtype=np.float32))
