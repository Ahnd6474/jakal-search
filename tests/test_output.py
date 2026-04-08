from __future__ import annotations

import json

import numpy as np

from jakal_search.config import EngineConfig
from jakal_search.engine import SearchTreeEngine
from jakal_search.output import render_json, render_records, render_report, render_tree, render_urls
from jakal_search.types import SearchDocument, SearchRequest
from jakal_search.utils import normalize_rows


def make_doc(title: str, source: str, query: str, published_at: str = "2026-04-07T12:00:00Z") -> SearchDocument:
    return SearchDocument(
        title=title,
        snippet=title,
        url=f"https://{source}/{title.replace(' ', '-')}",
        source=source,
        query=query,
        rank=1,
        published_at=published_at,
        published_at_precision="datetime",
    )


class FakeProvider:
    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        lowered = query.lower()
        if "policy" in lowered or "governance" in lowered:
            docs = POLICY_DOCS
        elif "tracking" in lowered or "vector" in lowered or "drift" in lowered:
            docs = TRACKING_DOCS
        elif "search system" in lowered:
            docs = ROOT_DOCS
        else:
            docs = []
        return docs[:max_results]


class KeywordEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(18, dtype=np.float32)
            if "search" in lowered:
                vector[0] += 1.0
            if "policy" in lowered or "governance" in lowered:
                vector[1] += 1.0
            if "tracking" in lowered or "vector" in lowered or "drift" in lowered:
                vector[2] += 1.0
            if "alignment" in lowered:
                vector[3] += 1.0
            if "miracle" in lowered or "conspiracy" in lowered or "secret" in lowered:
                vector[4] += 1.0
            for token in lowered.split():
                bucket = 5 + (sum(ord(char) for char in token) % 13)
                vector[bucket] += 0.15
            if not vector.any():
                vector[17] = 1.0
            rows.append(vector)
        return normalize_rows(np.asarray(rows, dtype=np.float32))


ROOT_DOCS = [
    make_doc("search policy governance roadmap", "arxiv.org", "search system"),
    make_doc("search policy standards", "acm.org", "search system"),
    make_doc("governance policy for search", "ieee.org", "search system"),
    make_doc("vector tracking for search branches", "github.com", "search system"),
    make_doc("tracking vector stability in search", "github.com", "search system"),
    make_doc("search branch vector drift", "openai.com", "search system"),
    make_doc("miracle secret search hack", "blogspot.com", "search system"),
    make_doc("conspiracy truth about search", "blogspot.com", "search system"),
]

POLICY_DOCS = [
    make_doc("policy governance for recursive search", "acm.org", "policy"),
    make_doc("policy evaluation for search systems", "ieee.org", "policy"),
    make_doc("governance process in exploratory search", "arxiv.org", "policy"),
    make_doc("search policy benchmark", "openai.com", "policy"),
]

TRACKING_DOCS = [
    make_doc("vector tracking and branch scoring", "github.com", "tracking"),
    make_doc("tracking stability for recursive search", "openai.com", "tracking"),
    make_doc("vector drift monitor for branches", "arxiv.org", "tracking"),
    make_doc("branch stability search metrics", "acm.org", "tracking"),
]


def make_tree(*, as_of: str | None = None):
    config = EngineConfig()
    config.limits.max_depth = 2
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.max_children_per_node = 2
    config.limits.results_per_query = 8
    config.similarity.dedupe_threshold = 0.995
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)
    if as_of is None:
        return engine.run("search system")
    return engine.run(SearchRequest(query="search system", metadata={"as_of": as_of}))


def test_render_report_groups_results_by_topic() -> None:
    report = render_report(make_tree())

    assert "Query: search system" in report
    assert "Summary" in report
    assert "Topics" in report
    assert "policy" in report.lower()
    assert "vector" in report.lower()
    assert "Sources" in report
    assert "evidence=" in report
    assert "retrieval=" in report


def test_render_urls_returns_unique_urls() -> None:
    urls = render_urls(make_tree()).splitlines()

    assert urls
    assert len(urls) == len(set(urls))
    assert any("acm.org" in url for url in urls)
    assert any("arxiv.org" in url for url in urls)


def test_render_tree_and_json_keep_existing_views() -> None:
    tree = make_tree()
    tree_output = render_tree(tree)
    json_output = render_json(tree)

    assert "- node-0: search system" in tree_output
    assert '"root_id": "node-0"' in json_output
    assert '"request"' in json_output
    assert '"expansions"' in json_output
    assert '"logs"' in json_output
    assert '"query": "search system"' in json_output
    assert '"document_id"' in json_output
    assert '"content"' in json_output
    assert '"embedding_dim"' in json_output
    assert '"source_profile"' in json_output
    assert '"topic"' in json_output
    assert '"theme_tokens"' in json_output
    assert '"outgoing_edges"' in json_output
    assert '"asset_condition"' in json_output
    assert '"candidate_scores"' in json_output


def test_render_records_emits_machine_readable_cutoff_fields() -> None:
    tree = make_tree(as_of="2026-04-08T09:00:00Z")
    records = json.loads(render_records(tree))

    assert records
    first = records[0]
    assert first["as_of"] == "2026-04-08T09:00:00Z"
    assert "published_at_precision" in first
    assert "lag_seconds" in first
    assert "dense_score" in first
    assert "lexical_score" in first
    assert "provider_score" in first
    assert "embedding_text" in first
    assert "theme_memberships" in first
    assert "topic_memberships" in first
    assert any(record["theme_memberships"] for record in records)
    assert any("base_score" in membership for record in records for membership in record["theme_memberships"])
    assert any(record["topic_memberships"] for record in records)
    assert any(
        "candidate_scores" in membership
        for record in records
        for membership in record["topic_memberships"]
    )
