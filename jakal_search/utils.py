from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" not in (parsed.netloc or ""):
        return url
    query = parse_qs(parsed.query)
    uddg = query.get("uddg")
    if not uddg:
        return url
    return unquote(uddg[0])


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype(np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.dot(left, right))


def mean_embedding(vectors: np.ndarray) -> np.ndarray | None:
    if vectors.size == 0:
        return None
    return normalize_vector(vectors.mean(axis=0))


def logistic(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))
