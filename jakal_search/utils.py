from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

DATE_PATTERNS = (
    re.compile(r"\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b"),
)
DATE_ONLY_PATTERNS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
)
DATETIME_PATTERNS = (
    "%Y%m%dT%H%M%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)


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


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


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


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, IndexError):
        pass
    for pattern in DATETIME_PATTERNS:
        try:
            return datetime.strptime(raw, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    for pattern in DATE_ONLY_PATTERNS:
        try:
            return datetime.strptime(raw, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    inferred = infer_date_from_text(raw)
    if inferred is None:
        return None
    try:
        return datetime.strptime(inferred, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def datetime_precision(value: str | None) -> str:
    if value is None:
        return "unknown"
    raw = value.strip()
    if not raw:
        return "unknown"
    lowered = raw.lower()
    if re.search(r"[t\s][0-2]\d:[0-5]\d", raw) or re.search(r"t\d{6}z?$", lowered):
        return "datetime"
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.hour or parsed.minute or parsed.second or parsed.tzinfo is not None:
            return "datetime"
    except (TypeError, ValueError, IndexError):
        pass
    if infer_date_from_text(raw) is not None:
        return "date"
    return "unknown"


def normalize_datetime_value(value: str | None) -> tuple[str | None, str]:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None, "unknown"
    precision = datetime_precision(value)
    return parsed.astimezone(UTC).isoformat(timespec="seconds"), precision


def infer_date_from_text(value: str | None) -> str | None:
    if value is None:
        return None
    for pattern in DATE_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 3:
            return f"{groups[0]}-{groups[1]}-{groups[2]}"
    return None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def age_in_days(value: str | None, *, now: datetime | None = None) -> float | None:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return max((current - parsed).total_seconds() / 86400.0, 0.0)


def is_at_or_before_as_of(
    published_at: str | None,
    as_of: str | None,
    *,
    published_precision: str = "unknown",
) -> bool:
    if not as_of:
        return True
    as_of_dt = parse_iso_datetime(as_of)
    if as_of_dt is None:
        return True
    published_dt = parse_iso_datetime(published_at)
    if published_dt is None:
        return False
    precision = published_precision if published_precision != "unknown" else datetime_precision(published_at)
    if precision == "date":
        as_of_precision = datetime_precision(as_of)
        if as_of_precision == "date":
            return published_dt.date() <= as_of_dt.date()
        return published_dt.date() < as_of_dt.date()
    return published_dt <= as_of_dt


def age_in_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return max((current - parsed).total_seconds(), 0.0)
