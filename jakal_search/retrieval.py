from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np
from bs4 import BeautifulSoup

from .types import PagePassage, SearchDocument
from .utils import (
    age_in_days,
    canonicalize_url,
    cosine_similarity,
    extract_domain,
    infer_date_from_text,
    jaccard_similarity,
    normalize_datetime_value,
    parse_iso_datetime,
    stable_hash,
    utc_now_iso,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-\._]{1,}", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_CONTENT_CHARS = 12000
DEFAULT_PASSAGE_WORDS = 90
DEFAULT_PASSAGE_STRIDE = 45
AFFILIATE_HINTS = ("affiliate", "ref=", "utm_", "sponsor", "sponsored")
DATE_META_KEYS = (
    "article:published_time",
    "article:modified_time",
    "og:updated_time",
    "last-modified",
    "date",
    "dc.date",
    "publishdate",
    "pubdate",
)
AUTHOR_META_KEYS = ("author", "article:author", "parsely-author")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if token and token.lower() not in STOPWORDS]


def build_passages(
    text: str,
    *,
    max_words: int = DEFAULT_PASSAGE_WORDS,
    stride_words: int = DEFAULT_PASSAGE_STRIDE,
) -> list[PagePassage]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [PagePassage(text=" ".join(words), start=0, end=len(words))]

    passages: list[PagePassage] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        passage_text = " ".join(words[start:end]).strip()
        if passage_text:
            passages.append(PagePassage(text=passage_text, start=start, end=end))
        if end >= len(words):
            break
        start += max(stride_words, 1)
    return passages


def passage_signature(text: str) -> str:
    tokens = tokenize_text(text)
    if not tokens:
        return ""
    counts = Counter(tokens)
    top_tokens = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[:12]
    return " ".join(token for token, _ in top_tokens)


def dedupe_passages(passages: list[PagePassage], *, threshold: float) -> list[PagePassage]:
    kept: list[PagePassage] = []
    token_sets: list[set[str]] = []
    for passage in passages:
        candidate_tokens = set(tokenize_text(passage.text))
        if not candidate_tokens:
            continue
        if any(jaccard_similarity(candidate_tokens, existing_tokens) >= threshold for existing_tokens in token_sets):
            continue
        kept.append(passage)
        token_sets.append(candidate_tokens)
    return kept


def extract_readable_payload(html: str, *, url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    head = soup.head
    metadata: dict[str, object] = {}

    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        node.decompose()

    title_node = soup.find("title")
    title = title_node.get_text(" ", strip=True) if title_node else ""

    if head is not None:
        for meta in head.find_all("meta"):
            name = str(meta.get("name") or meta.get("property") or "").strip().lower()
            content = str(meta.get("content") or "").strip()
            if not name or not content:
                continue
            if name in DATE_META_KEYS and "published_at" not in metadata:
                metadata["published_at"] = content
            if name in AUTHOR_META_KEYS and "author" not in metadata:
                metadata["author"] = content
            if name in {"description", "og:description"} and "description" not in metadata:
                metadata["description"] = content

    candidates = soup.select("article, main")
    if candidates:
        body = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    else:
        body = soup.body or soup

    text = body.get_text("\n", strip=True)
    paragraphs = [segment.strip() for segment in SENTENCE_PATTERN.split(text) if segment and len(segment.strip()) >= 40]
    collapsed = " ".join(paragraphs)
    if len(collapsed) > MAX_CONTENT_CHARS:
        collapsed = collapsed[:MAX_CONTENT_CHARS]

    links = [str(link.get("href") or "").strip() for link in body.find_all("a", href=True)]
    outbound_links = [link for link in links if link.startswith(("http://", "https://")) and extract_domain(link) != extract_domain(url)]
    about_links = [link for link in links if any(token in link.lower() for token in ("/about", "/contact", "about", "contact"))]
    citation_count = sum(
        1
        for paragraph in paragraphs
        if any(marker in paragraph.lower() for marker in ("doi", "citation", "references", "[1]", "(202", "et al"))
    )
    affiliate_link_count = sum(1 for link in links if any(marker in link.lower() for marker in AFFILIATE_HINTS))
    duplicate_ratio = _estimate_duplicate_ratio(paragraphs)
    published_at, published_precision = normalize_datetime_value(
        str(metadata.get("published_at") or "") or infer_date_from_text(url) or infer_date_from_text(collapsed)
    )
    if published_at:
        metadata["published_at"] = published_at
        metadata["published_at_precision"] = published_precision

    metadata.update(
        {
            "outbound_link_count": len(outbound_links),
            "citation_count": citation_count,
            "affiliate_link_count": affiliate_link_count,
            "has_about_link": bool(about_links),
            "duplicate_ratio": duplicate_ratio,
            "text_length": len(collapsed),
            "content_hash": stable_hash(collapsed) if collapsed else "",
            "canonical_url": canonicalize_url(url),
            "fetched_at": utc_now_iso(),
        }
    )
    return {"title": title, "text": collapsed, "metadata": metadata}


def _estimate_duplicate_ratio(paragraphs: list[str]) -> float:
    if not paragraphs:
        return 0.0
    normalized = [" ".join(paragraph.lower().split()) for paragraph in paragraphs if paragraph.strip()]
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / max(len(normalized), 1)


class PageContentFetcher:
    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_concurrency: int = 6,
        per_host_delay_seconds: float = 0.4,
        cache_dir: str | Path = "outputs/cache/pages",
        enable_cache: bool = True,
        cache_ttl_hours: int = 72,
    ) -> None:
        self._timeout = timeout
        self._max_concurrency = max(max_concurrency, 1)
        self._per_host_delay_seconds = max(per_host_delay_seconds, 0.0)
        self._enable_cache = enable_cache
        self._cache_ttl_seconds = max(cache_ttl_hours, 1) * 3600
        self._cache_dir = Path(cache_dir)
        if self._enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._host_last_request: defaultdict[str, float] = defaultdict(float)
        self._host_lock: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        if max_docs <= 0 or not docs:
            return docs
        return asyncio.run(self.enrich_documents_async(docs, max_docs=max_docs))

    async def enrich_documents_async(self, docs: list[SearchDocument], *, max_docs: int) -> list[SearchDocument]:
        semaphore = asyncio.Semaphore(self._max_concurrency)
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=self._timeout,
            headers={"User-Agent": "jakal-search/0.1"},
        ) as client:
            tasks = [
                self._enrich_one(client, semaphore, doc.clone(), index < max_docs)
                for index, doc in enumerate(docs)
            ]
            return list(await asyncio.gather(*tasks))

    async def _enrich_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        doc: SearchDocument,
        should_fetch: bool,
    ) -> SearchDocument:
        if doc.content and len(doc.content) > len(doc.snippet) + 80:
            if not doc.passages:
                doc.passages = build_passages(doc.content)
            return doc
        if not should_fetch or not doc.url.startswith(("http://", "https://")):
            if not doc.passages and doc.content:
                doc.passages = build_passages(doc.content)
            return doc

        cached_payload = self._load_cache(doc.url)
        if cached_payload is not None:
            self._apply_cached_payload(doc, cached_payload)
            return doc

        try:
            async with semaphore:
                await self._respect_rate_limit(doc.url)
                response = await client.get(doc.url)
                response.raise_for_status()
        except Exception:
            if not doc.passages and doc.content:
                doc.passages = build_passages(doc.content)
            return doc

        payload = extract_readable_payload(response.text, url=doc.url)
        title = str(payload.get("title") or "").strip()
        text = str(payload.get("text") or "").strip()
        metadata = dict(payload.get("metadata") or {})
        if title and len(title) > len(doc.title):
            doc.title = title
        if text:
            doc.content = text
        doc.passages = build_passages(doc.content)
        doc.published_at, doc.published_at_precision = _normalize_published_at(metadata.get("published_at"), doc.url, doc.content)
        if doc.published_at is not None:
            metadata["published_at"] = doc.published_at
            metadata["published_at_precision"] = doc.published_at_precision
        doc.metadata.update(metadata)
        if self._enable_cache:
            self._save_cache(doc)
        return doc

    async def _respect_rate_limit(self, url: str) -> None:
        host = extract_domain(url)
        if not host or self._per_host_delay_seconds <= 0.0:
            return
        async with self._host_lock[host]:
            now = time.monotonic()
            previous = self._host_last_request[host]
            remaining = self._per_host_delay_seconds - (now - previous)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._host_last_request[host] = time.monotonic()

    def _cache_path(self, url: str) -> Path:
        return self._cache_dir / f"{stable_hash(canonicalize_url(url))}.json"

    def _load_cache(self, url: str) -> dict[str, object] | None:
        if not self._enable_cache:
            return None
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        cached_at = age_in_days(str(payload.get("cached_at") or ""), now=None)
        if cached_at is not None and (cached_at * 86400.0) > self._cache_ttl_seconds:
            return None
        return payload

    def _save_cache(self, doc: SearchDocument) -> None:
        if not self._enable_cache:
            return
        payload = {
            "cached_at": utc_now_iso(),
            "title": doc.title,
            "content": doc.content,
            "passages": [passage.to_dict() for passage in doc.passages],
            "published_at": doc.published_at,
            "published_at_precision": doc.published_at_precision,
            "metadata": doc.metadata,
        }
        self._cache_path(doc.url).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _apply_cached_payload(self, doc: SearchDocument, payload: dict[str, object]) -> None:
        doc.title = str(payload.get("title") or doc.title)
        doc.content = str(payload.get("content") or doc.content)
        doc.passages = [PagePassage(**row) for row in list(payload.get("passages") or [])]
        doc.published_at = payload.get("published_at") if isinstance(payload.get("published_at"), str) else doc.published_at
        doc.published_at_precision = (
            payload.get("published_at_precision") if isinstance(payload.get("published_at_precision"), str) else doc.published_at_precision
        )
        doc.metadata.update(dict(payload.get("metadata") or {}))
        if not doc.passages and doc.content:
            doc.passages = build_passages(doc.content)


def _normalize_published_at(value: object, url: str, content: str) -> tuple[str | None, str]:
    normalized_value, precision = normalize_datetime_value(value if isinstance(value, str) else None)
    if normalized_value is not None:
        return normalized_value, precision
    return normalize_datetime_value(infer_date_from_text(url) or infer_date_from_text(content))


class HybridRetrievalRanker:
    def __init__(
        self,
        *,
        lexical_weight: float = 0.35,
        dense_weight: float = 0.4,
        provider_weight: float = 0.15,
        domain_weight: float = 0.1,
        freshness_weight: float = 0.12,
        entity_weight: float = 0.08,
        max_passages: int = 3,
        passage_dedupe_threshold: float = 0.82,
    ) -> None:
        self._lexical_weight = lexical_weight
        self._dense_weight = dense_weight
        self._provider_weight = provider_weight
        self._domain_weight = domain_weight
        self._freshness_weight = freshness_weight
        self._entity_weight = entity_weight
        self._max_passages = max_passages
        self._passage_dedupe_threshold = passage_dedupe_threshold

    def rank(
        self,
        *,
        query: str,
        query_vector: np.ndarray,
        docs: list[SearchDocument],
        entity_hints: list[str] | None = None,
        freshness_required: bool = False,
        as_of: str | None = None,
    ) -> list[SearchDocument]:
        if not docs:
            return []

        query_terms = tokenize_text(query)
        doc_term_counts = [Counter(tokenize_text(doc.text)) for doc in docs]
        document_frequency = Counter()
        for counts in doc_term_counts:
            document_frequency.update(counts.keys())
        average_doc_len = (
            float(np.mean([max(sum(counts.values()), 1) for counts in doc_term_counts]))
            if doc_term_counts
            else 1.0
        )

        ranked: list[SearchDocument] = []
        for doc, term_counts in zip(docs, doc_term_counts):
            lexical = self._bm25_lite(query_terms, term_counts, document_frequency, len(docs), average_doc_len)
            dense = max(0.0, cosine_similarity(query_vector, doc.embedding))
            provider = float(doc.metadata.get("provider_score", 1.0 / max(doc.rank, 1)))
            domain = 0.5
            profile = doc.source_profile
            if profile is not None:
                domain = float(profile.domain_score)
            elif "domain_score" in doc.metadata:
                domain = float(doc.metadata["domain_score"])
            freshness = self._freshness_score(doc.published_at, freshness_required=freshness_required, as_of=as_of)
            entity = self._entity_score(doc, entity_hints or [])

            retrieval_score = (
                (self._lexical_weight * lexical)
                + (self._dense_weight * dense)
                + (self._provider_weight * provider)
                + (self._domain_weight * domain)
                + (self._freshness_weight * freshness)
                + (self._entity_weight * entity)
            )
            doc.lexical_score = float(lexical)
            doc.dense_score = float(dense)
            doc.provider_score = float(provider)
            doc.freshness_score = float(freshness)
            doc.entity_score = float(entity)
            doc.retrieval_score = float(retrieval_score)
            doc.passages = self._score_passages(query_terms, doc, entity_hints or [])
            ranked.append(doc)

        ranked.sort(
            key=lambda item: (
                item.retrieval_score,
                item.trust_score,
                item.freshness_score,
                -item.rank,
                item.title.lower(),
            ),
            reverse=True,
        )
        return ranked

    def _score_passages(self, query_terms: list[str], doc: SearchDocument, entity_hints: list[str]) -> list[PagePassage]:
        passages = list(doc.passages)
        if not passages and doc.content:
            passages = build_passages(doc.content)
        if not passages:
            text = doc.content or doc.snippet or doc.title
            if text:
                passages = build_passages(text)
        scored: list[PagePassage] = []
        for passage in dedupe_passages(passages, threshold=self._passage_dedupe_threshold):
            terms = Counter(tokenize_text(passage.text))
            lexical = self._bm25_lite(query_terms, terms, Counter(terms.keys()), 1, max(sum(terms.values()), 1))
            entity_bonus = 0.0
            lowered = passage.text.lower()
            if entity_hints:
                matches = sum(1 for hint in entity_hints if hint and hint in lowered)
                entity_bonus = matches / max(len(entity_hints), 1)
            passage.score = float(lexical + (0.2 * entity_bonus))
            scored.append(passage)
        scored.sort(key=lambda item: (item.score, len(item.text)), reverse=True)
        return scored[: self._max_passages]

    def _freshness_score(self, published_at: str | None, *, freshness_required: bool, as_of: str | None) -> float:
        reference_time = parse_iso_datetime(as_of)
        age_days = age_in_days(published_at, now=reference_time)
        if age_days is None:
            return 0.0 if freshness_required else 0.12
        if freshness_required:
            return math.exp(-age_days / 30.0)
        return math.exp(-age_days / 365.0)

    def _entity_score(self, doc: SearchDocument, entity_hints: list[str]) -> float:
        if not entity_hints:
            return 0.0
        lowered = doc.text.lower()
        matches = sum(1 for hint in entity_hints if hint and hint in lowered)
        return matches / max(len(entity_hints), 1)

    def _bm25_lite(
        self,
        query_terms: list[str],
        doc_counts: Counter[str],
        document_frequency: Counter[str],
        corpus_size: int,
        average_doc_len: float,
    ) -> float:
        if not query_terms:
            return 0.0
        doc_len = sum(doc_counts.values())
        if doc_len == 0:
            return 0.0

        score = 0.0
        k1 = 1.2
        b = 0.75
        for term in query_terms:
            term_frequency = doc_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            df = max(document_frequency.get(term, 0), 1)
            idf = math.log(1.0 + ((corpus_size - df + 0.5) / (df + 0.5)))
            numerator = term_frequency * (k1 + 1.0)
            denominator = term_frequency + k1 * (1.0 - b + (b * (doc_len / max(average_doc_len, 1.0))))
            score += idf * (numerator / max(denominator, 1e-6))
        return float(score)
