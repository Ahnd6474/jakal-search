from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]{1,}")
PHRASE_PATTERN = re.compile(r'"([^"]+)"')
VERSION_PATTERN = re.compile(r"\b[a-zA-Z]+[-\s]?\d+(?:\.\d+){0,2}\b")
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")

NEWS_HINTS = {"latest", "recent", "today", "news", "current", "update", "updated", "breaking"}
COMPARATIVE_HINTS = {"vs", "versus", "compare", "comparison", "best", "better", "difference", "alternative"}
TROUBLESHOOT_HINTS = {"error", "fix", "debug", "issue", "problem", "traceback", "failing", "failure", "bug"}
TECHNICAL_HINTS = {"api", "sdk", "python", "javascript", "typescript", "docs", "documentation", "package", "library"}
RESEARCH_HINTS = {"paper", "research", "study", "arxiv", "benchmark", "evaluation", "dataset", "regression"}
NAVIGATIONAL_HINTS = {"official", "homepage", "github", "docs", "website"}
MARKET_HINTS = {
    "stock",
    "stocks",
    "share",
    "shares",
    "equity",
    "earnings",
    "guidance",
    "forecast",
    "analyst",
    "downgrade",
    "upgrade",
    "sec",
    "revenue",
    "profit",
    "quarterly",
    "market",
}

ENTITY_STOPWORDS = {
    "best",
    "compare",
    "difference",
    "docs",
    "documentation",
    "error",
    "fix",
    "guide",
    "how",
    "latest",
    "news",
    "recent",
    "tutorial",
    "vs",
}


@dataclass(slots=True)
class QueryAnalysis:
    query: str
    normalized_query: str
    tokens: list[str]
    intent: str
    domain_pack: str
    freshness_required: bool
    troubleshooting: bool = False
    comparative: bool = False
    navigational: bool = False
    market_news: bool = False
    entity_hints: list[str] = field(default_factory=list)
    ticker_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "tokens": self.tokens,
            "intent": self.intent,
            "domain_pack": self.domain_pack,
            "freshness_required": self.freshness_required,
            "troubleshooting": self.troubleshooting,
            "comparative": self.comparative,
            "navigational": self.navigational,
            "market_news": self.market_news,
            "entity_hints": self.entity_hints,
            "ticker_hints": self.ticker_hints,
        }


def analyze_query(query: str) -> QueryAnalysis:
    normalized_query = " ".join(query.strip().split())
    tokens = [token.lower() for token in QUERY_TOKEN_PATTERN.findall(normalized_query)]
    token_set = set(tokens)
    freshness_required = bool(token_set & NEWS_HINTS) or bool(re.search(r"\b20\d{2}\b", normalized_query))
    comparative = bool(token_set & COMPARATIVE_HINTS)
    troubleshooting = bool(token_set & TROUBLESHOOT_HINTS)
    navigational = bool(token_set & NAVIGATIONAL_HINTS) and len(tokens) <= 6
    ticker_hints = extract_ticker_hints(query)
    market_news = bool(token_set & MARKET_HINTS) or bool(ticker_hints)
    if market_news:
        freshness_required = True

    if market_news:
        intent = "market_news"
    elif freshness_required:
        intent = "news"
    elif troubleshooting:
        intent = "troubleshooting"
    elif comparative:
        intent = "comparative"
    elif token_set & RESEARCH_HINTS:
        intent = "research"
    elif navigational:
        intent = "navigational"
    elif token_set & TECHNICAL_HINTS:
        intent = "technical"
    else:
        intent = "exploratory"

    if intent == "market_news":
        domain_pack = "market_news"
    elif intent == "research":
        domain_pack = "research"
    elif intent in {"technical", "troubleshooting", "navigational"}:
        domain_pack = "technical"
    elif intent == "news":
        domain_pack = "news"
    else:
        domain_pack = "general"

    entity_hints = list(dict.fromkeys([*ticker_hints, *extract_entity_hints(normalized_query)]))
    return QueryAnalysis(
        query=query,
        normalized_query=normalized_query.lower(),
        tokens=tokens,
        intent=intent,
        domain_pack=domain_pack,
        freshness_required=freshness_required,
        troubleshooting=troubleshooting,
        comparative=comparative,
        navigational=navigational,
        market_news=market_news,
        entity_hints=entity_hints,
        ticker_hints=ticker_hints,
    )


def extract_entity_hints(query: str) -> list[str]:
    candidates: Counter[str] = Counter()
    lowered = query.lower()
    for phrase in PHRASE_PATTERN.findall(query):
        normalized = " ".join(phrase.lower().split())
        if normalized:
            candidates[normalized] += 4

    for match in VERSION_PATTERN.findall(query):
        normalized = " ".join(match.lower().split())
        if normalized:
            candidates[normalized] += 3

    tokens = [token.lower() for token in QUERY_TOKEN_PATTERN.findall(query)]
    for size in (3, 2, 1):
        for index in range(0, max(len(tokens) - size + 1, 0)):
            phrase = " ".join(tokens[index : index + size])
            if not phrase or phrase in ENTITY_STOPWORDS:
                continue
            parts = phrase.split()
            if any(part in ENTITY_STOPWORDS for part in parts):
                continue
            if size == 1 and len(parts[0]) < 4:
                continue
            score = size + (2 if any(char.isdigit() for char in phrase) else 0)
            if phrase in lowered:
                candidates[phrase] += score

    ranked = [phrase for phrase, _ in candidates.most_common(8)]
    return ranked


def rank_document_entities(query_analysis: QueryAnalysis, texts: list[str]) -> list[str]:
    if not texts:
        return list(query_analysis.entity_hints)
    candidates = Counter(query_analysis.entity_hints)
    for text in texts:
        lowered = text.lower()
        for hint in query_analysis.entity_hints:
            if hint in lowered:
                candidates[hint] += 2
    return [entity for entity, _ in candidates.most_common(6)]


def extract_ticker_hints(query: str) -> list[str]:
    tickers = [match.group(0) for match in TICKER_PATTERN.finditer(query) if match.group(0).isupper()]
    seen: list[str] = []
    for ticker in tickers:
        lowered = ticker.lower()
        if lowered in {"api", "sdk", "usa"}:
            continue
        if ticker not in seen:
            seen.append(ticker)
    return seen[:4]
