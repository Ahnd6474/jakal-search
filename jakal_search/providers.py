from __future__ import annotations

import html
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Protocol
from urllib.parse import quote

import httpx

from .query_analysis import analyze_query
from .retrieval import PageContentFetcher
from .types import SearchDocument
from .utils import canonicalize_url, extract_domain, normalize_datetime_value, unwrap_duckduckgo_url

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _published_fields(raw_value: object) -> tuple[str | None, str]:
    if not isinstance(raw_value, str):
        return None, "unknown"
    return normalize_datetime_value(raw_value)


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        ...

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        ...


class DuckDuckGoHtmlProvider:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(
            follow_redirects=True,
            headers={"User-Agent": "jakal-search/0.1"},
            timeout=timeout,
        )
        self._search_url = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        from bs4 import BeautifulSoup

        response = self._client.get(self._search_url, params={"q": query})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        docs: list[SearchDocument] = []

        for rank, block in enumerate(soup.select("div.result"), start=1):
            link = block.select_one("a.result__a")
            if link is None:
                continue
            snippet_node = block.select_one(".result__snippet")
            raw_url = str(link.get("href", "")).strip()
            if not raw_url:
                continue
            url = unwrap_duckduckgo_url(raw_url)
            docs.append(
                SearchDocument(
                    title=link.get_text(" ", strip=True),
                    snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    url=url,
                    source=extract_domain(url),
                    query=query,
                    rank=rank,
                    metadata={"provider": "duckduckgo_html", "provider_rank": rank},
                )
            )
            if len(docs) >= max_results:
                break
        return docs

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        return docs


class WikipediaSearchProvider:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "jakal-search/0.1"})
        self._search_url = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        response = self._client.get(
            self._search_url,
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "utf8": 1,
                "srlimit": max_results,
                "srsearch": query,
            },
        )
        response.raise_for_status()
        payload = response.json()
        docs: list[SearchDocument] = []
        for rank, row in enumerate(payload.get("query", {}).get("search", []), start=1):
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            snippet = html.unescape(str(row.get("snippet") or "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""))
            url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            published_at, precision = _published_fields(row.get("timestamp"))
            docs.append(
                SearchDocument(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source="wikipedia.org",
                    query=query,
                    rank=rank,
                    published_at=published_at,
                    published_at_precision=precision,
                    metadata={"provider": "wikipedia", "provider_rank": rank, "published_at_precision": precision},
                )
            )
        return docs

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        return docs


class ArxivApiProvider:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "jakal-search/0.1"})
        self._search_url = "https://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        response = self._client.get(
            self._search_url,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
            },
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        docs: list[SearchDocument] = []
        for rank, entry in enumerate(root.findall("atom:entry", ATOM_NS), start=1):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split())
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").split())
            link = entry.find("atom:id", ATOM_NS)
            url = link.text.strip() if link is not None and link.text else ""
            if not title or not url:
                continue
            published_at, precision = _published_fields(entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or None)
            docs.append(
                SearchDocument(
                    title=title,
                    snippet=summary,
                    url=url,
                    source="arxiv.org",
                    query=query,
                    rank=rank,
                    published_at=published_at,
                    published_at_precision=precision,
                    metadata={"provider": "arxiv", "provider_rank": rank, "published_at_precision": precision},
                )
            )
        return docs

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        return docs


class HackerNewsAlgoliaProvider:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "jakal-search/0.1"})
        self._search_url = "https://hn.algolia.com/api/v1/search"

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        response = self._client.get(
            self._search_url,
            params={"query": query, "hitsPerPage": max_results},
        )
        response.raise_for_status()
        payload = response.json()
        docs: list[SearchDocument] = []
        for rank, hit in enumerate(payload.get("hits", []), start=1):
            url = str(hit.get("url") or hit.get("story_url") or "").strip()
            if not url:
                continue
            title = str(hit.get("title") or hit.get("story_title") or "").strip()
            if not title:
                continue
            snippet = str(hit.get("story_text") or hit.get("comment_text") or "").strip()
            published_at, precision = _published_fields(hit.get("created_at"))
            docs.append(
                SearchDocument(
                    title=title,
                    snippet=snippet,
                    url=url,
                    source=extract_domain(url),
                    query=query,
                    rank=rank,
                    published_at=published_at,
                    published_at_precision=precision,
                    metadata={"provider": "hackernews", "provider_rank": rank, "published_at_precision": precision},
                )
            )
        return docs

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        return docs


class GoogleNewsRssProvider:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "jakal-search/0.1"})
        self._search_url = "https://news.google.com/rss/search"

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        response = self._client.get(self._search_url, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        response.raise_for_status()
        root = ET.fromstring(response.text)
        docs: list[SearchDocument] = []
        for rank, item in enumerate(root.findall("./channel/item"), start=1):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            if not title or not url:
                continue
            source = extract_domain(url) or "news.google.com"
            source_node = item.find("source")
            if source_node is not None and source_node.text:
                source = source_node.text.strip().lower()
            published_at, precision = _published_fields(item.findtext("pubDate"))
            docs.append(
                SearchDocument(
                    title=title,
                    snippet=(item.findtext("description") or "").strip(),
                    url=url,
                    source=source,
                    query=query,
                    rank=rank,
                    published_at=published_at,
                    published_at_precision=precision,
                    metadata={"provider": "google_news_rss", "provider_rank": rank, "published_at_precision": precision},
                )
            )
            if len(docs) >= max_results:
                break
        return docs

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        return docs


class GdeltNewsProvider:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "jakal-search/0.1"})
        self._search_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        response = self._client.get(
            self._search_url,
            params={
                "query": query,
                "mode": "ArtList",
                "maxrecords": max_results,
                "format": "json",
                "sort": "HybridRel",
            },
        )
        response.raise_for_status()
        payload = response.json()
        docs: list[SearchDocument] = []
        for rank, article in enumerate(payload.get("articles", []), start=1):
            url = str(article.get("url") or "").strip()
            title = str(article.get("title") or "").strip()
            if not title or not url:
                continue
            source = str(article.get("sourceCommonName") or extract_domain(url)).strip().lower()
            published_at, precision = _published_fields(article.get("seendate"))
            docs.append(
                SearchDocument(
                    title=title,
                    snippet=str(article.get("seendate") or "").strip(),
                    url=url,
                    source=source,
                    query=query,
                    rank=rank,
                    published_at=published_at,
                    published_at_precision=precision,
                    metadata={"provider": "gdelt", "provider_rank": rank, "published_at_precision": precision},
                )
            )
        return docs

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        return docs


class MultiSearchProvider:
    def __init__(
        self,
        *,
        timeout: float = 15.0,
        fetch_page_content: bool = True,
        provider_pack: str = "auto",
        fetcher: PageContentFetcher | None = None,
    ) -> None:
        self._provider_pack = provider_pack
        self._providers = {
            "general": [DuckDuckGoHtmlProvider(timeout=timeout), WikipediaSearchProvider(timeout=timeout)],
            "technical": [
                DuckDuckGoHtmlProvider(timeout=timeout),
                HackerNewsAlgoliaProvider(timeout=timeout),
                WikipediaSearchProvider(timeout=timeout),
            ],
            "research": [
                DuckDuckGoHtmlProvider(timeout=timeout),
                ArxivApiProvider(timeout=timeout),
                WikipediaSearchProvider(timeout=timeout),
            ],
            "news": [
                DuckDuckGoHtmlProvider(timeout=timeout),
                HackerNewsAlgoliaProvider(timeout=timeout),
                WikipediaSearchProvider(timeout=timeout),
            ],
            "market_news": [
                GoogleNewsRssProvider(timeout=timeout),
                GdeltNewsProvider(timeout=timeout),
                DuckDuckGoHtmlProvider(timeout=timeout),
            ],
        }
        self._fetcher = fetcher if fetcher is not None else (
            PageContentFetcher(timeout=max(timeout, 10.0)) if fetch_page_content else None
        )

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        analysis = analyze_query(query)
        pack = analysis.domain_pack if self._provider_pack == "auto" else self._provider_pack
        providers = self._providers.get(pack, self._providers["general"])
        per_provider_limit = max(math.ceil(max_results * 0.75), 4)
        result_sets: list[list[SearchDocument]] = []
        for provider in providers:
            try:
                result_sets.append(provider.search(query, per_provider_limit))
            except Exception:
                continue
        return fuse_provider_results(result_sets, max_results=max_results)

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        if self._fetcher is None:
            return docs
        return self._fetcher.enrich_documents(docs, max_docs=max_docs)


def fuse_provider_results(result_sets: list[list[SearchDocument]], *, max_results: int) -> list[SearchDocument]:
    fused: dict[str, SearchDocument] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    providers_by_url: defaultdict[str, set[str]] = defaultdict(set)
    for result_set in result_sets:
        for rank, doc in enumerate(result_set, start=1):
            url = canonicalize_url(doc.url)
            rrf = 1.0 / (60.0 + rank)
            scores[url] += rrf
            providers_by_url[url].add(str(doc.metadata.get("provider") or "unknown"))
            existing = fused.get(url)
            if existing is None:
                clone = doc.clone()
                clone.url = url
                fused[url] = clone
                continue
            if len(doc.snippet) > len(existing.snippet):
                existing.snippet = doc.snippet
            if doc.published_at and not existing.published_at:
                existing.published_at = doc.published_at
                existing.published_at_precision = doc.published_at_precision
            existing.metadata.update(doc.metadata)

    ranked = sorted(
        fused.values(),
        key=lambda item: (scores[item.url], -item.rank, item.title.lower()),
        reverse=True,
    )[:max_results]
    for rank, doc in enumerate(ranked, start=1):
        doc.rank = rank
        doc.metadata["provider_names"] = sorted(providers_by_url[doc.url])
        doc.metadata["provider_score"] = float(scores[doc.url])
    return ranked
