from __future__ import annotations

from typing import Protocol

import httpx
from bs4 import BeautifulSoup

from .types import SearchDocument
from .utils import extract_domain, unwrap_duckduckgo_url


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchDocument]:
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
            source = extract_domain(url)
            docs.append(
                SearchDocument(
                    title=link.get_text(" ", strip=True),
                    snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    url=url,
                    source=source,
                    query=query,
                    rank=rank,
                )
            )
            if len(docs) >= max_results:
                break
        return docs
