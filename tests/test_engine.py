from __future__ import annotations

import numpy as np

from jakal_search.config import EngineConfig
from jakal_search.engine import SearchTreeEngine, SimilarityDeduper
from jakal_search.types import DocumentMemoryItem, SearchDocument, SearchRequest
from jakal_search.utils import normalize_rows


def make_doc(title: str, source: str, query: str) -> SearchDocument:
    return SearchDocument(
        title=title,
        snippet=title,
        url=f"https://{source}/{title.replace(' ', '-')}",
        source=source,
        query=query,
        rank=1,
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

    def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
        enriched: list[SearchDocument] = []
        for index, doc in enumerate(docs):
            clone = doc.clone()
            if index < max_docs:
                clone.content = (
                    f"{clone.title}. "
                    f"{clone.title} detailed evidence and supporting explanation for {clone.query}. "
                    f"Additional context about {clone.source} and reproducible validation steps."
                )
            enriched.append(clone)
        return enriched


class KeywordEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(18, dtype=np.float32)
            if "search" in lowered:
                vector[0] += 0.6
            if "policy" in lowered or "governance" in lowered:
                vector[1] += 2.0
            if "tracking" in lowered or "vector" in lowered or "drift" in lowered:
                vector[2] += 2.0
            if "alignment" in lowered:
                vector[3] += 1.0
            if "miracle" in lowered or "conspiracy" in lowered or "secret" in lowered:
                vector[4] += 1.5
            for token in lowered.split():
                bucket = 5 + (sum(ord(char) for char in token) % 13)
                vector[bucket] += 0.05
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


def test_deduper_keeps_parent_similar_docs_but_blocks_external_duplicates() -> None:
    deduper = SimilarityDeduper(threshold=0.9)
    docs = [
        make_doc("policy item", "acm.org", "root"),
        make_doc("policy copy", "acm.org", "root"),
    ]
    embeddings = normalize_rows(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    for doc, embedding in zip(docs, embeddings):
        doc.embedding = embedding
    memory = [
        DocumentMemoryItem(
            node_id="parent",
            document_id="https://acm.org/parent",
            url="https://acm.org/parent",
            embedding=embeddings[0],
        ),
        DocumentMemoryItem(
            node_id="sibling",
            document_id="https://acm.org/sibling",
            url="https://acm.org/sibling",
            embedding=embeddings[1],
        ),
    ]

    kept_docs = deduper.dedupe(
        docs=docs,
        ancestry_ids={"parent", "current"},
        memory=memory,
    )

    assert len(kept_docs) == 1
    assert kept_docs[0].title == "policy item"


def test_engine_builds_subtopics_and_filters_low_trust_noise() -> None:
    config = EngineConfig()
    config.limits.max_depth = 2
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.max_children_per_node = 2
    config.limits.results_per_query = 8
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)
    tree = engine.run("search system")

    root = tree.nodes[tree.root_id]
    child_queries = {tree.nodes[child_id].query for child_id in root.children}

    assert root.status == "expanded"
    assert len(root.children) == 2
    assert any("policy" in query for query in child_queries)
    assert any("vector" in query or "tracking" in query for query in child_queries)
    assert all("miracle" not in doc.title for doc in root.docs)
    assert all(doc.embedding is not None for doc in root.docs)
    assert all(doc.content for doc in root.docs)
    assert all(doc.passages for doc in root.docs)
    assert all(doc.retrieval_score >= 0.0 for doc in root.docs)
    assert all(doc.document_id for doc in root.docs)
    assert all(doc.source_profile is not None for doc in root.docs)
    assert root.theme_tokens
    assert all(doc.metadata.get("theme_memberships") for doc in root.docs)
    assert all(token.outgoing_edges for token in root.theme_tokens)
    assert all(token.asset_condition for token in root.theme_tokens)
    assert all(token.query for token in root.theme_tokens)
    assert all(token.candidate_scores for token in root.theme_tokens)
    assert all(
        edge["target_token_id"] != token.token_id
        for token in root.theme_tokens
        for edge in token.outgoing_edges
    )
    assert all(
        sum(abs(float(edge["weight"])) for edge in token.outgoing_edges) <= 1.00001
        for token in root.theme_tokens
    )
    assert any(doc.source_profile.source_type == "research" for doc in root.docs if doc.source_profile)
    assert all("decision_reason" in tree.nodes[child_id].metrics for child_id in root.children)
    assert all("evidence_coverage" in tree.nodes[child_id].metrics for child_id in root.children)
    assert all(tree.nodes[child_id].theme_tokens for child_id in root.children)
    assert all(tree.nodes[child_id].topic is not None for child_id in root.children)
    assert tree.expansions[root.node_id].raw_document_ids
    assert tree.expansions[root.node_id].trusted_document_ids
    assert tree.expansions[root.node_id].unique_document_ids
    assert tree.logs
    assert any(event.event_type == "trust_filter" for event in tree.logs)


def test_max_depth_stops_expansion() -> None:
    config = EngineConfig()
    config.limits.max_depth = 1
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.results_per_query = 8
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)
    tree = engine.run("search system")

    for child_id in tree.nodes[tree.root_id].children:
        assert tree.nodes[child_id].depth == 1
        assert tree.nodes[child_id].status in {"pending", "pruned", "stopped"}


def test_engine_reuse_resets_run_state() -> None:
    config = EngineConfig()
    config.limits.max_depth = 2
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.max_children_per_node = 2
    config.limits.results_per_query = 8
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)

    first_tree = engine.run("search system")
    second_tree = engine.run("search system")

    first_root = first_tree.nodes[first_tree.root_id]
    second_root = second_tree.nodes[second_tree.root_id]
    first_queries = {first_tree.nodes[child_id].query for child_id in first_root.children}
    second_queries = {second_tree.nodes[child_id].query for child_id in second_root.children}

    assert first_tree.root_id == "node-0"
    assert second_tree.root_id == "node-0"
    assert first_root.status == "expanded"
    assert second_root.status == "expanded"
    assert first_root.stop_reason is None
    assert second_root.stop_reason is None
    assert len(first_root.children) == 2
    assert len(second_root.children) == 2
    assert first_queries == second_queries


def test_document_clone_prevents_provider_state_leakage() -> None:
    doc = ROOT_DOCS[0]
    original_embedding = doc.embedding

    config = EngineConfig()
    config.limits.max_depth = 1
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.results_per_query = 8
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)
    engine.run("search system")

    assert ROOT_DOCS[0].embedding is original_embedding
    assert ROOT_DOCS[0].cluster_id is None


def test_search_request_overrides_limits_without_mutating_engine_defaults() -> None:
    config = EngineConfig()
    config.limits.max_depth = 3
    config.limits.max_total_nodes = 24
    config.limits.frontier_width = 6
    config.limits.results_per_query = 8
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)
    request = SearchRequest(
        query="search system",
        max_depth=1,
        max_total_nodes=3,
        frontier_width=2,
        results_per_query=6,
        metadata={"suite": "engine"},
    )
    tree = engine.run(request)

    assert tree.request is not None
    assert tree.request.max_depth == 1
    assert tree.request.metadata["suite"] == "engine"
    assert engine.config.limits.max_depth == 3
    assert engine.config.limits.max_total_nodes == 24
    assert len(tree.nodes) <= 3
    assert all(tree.nodes[child_id].depth == 1 for child_id in tree.nodes[tree.root_id].children)


def test_expansion_context_tracks_stop_reason_for_terminal_nodes() -> None:
    config = EngineConfig()
    config.limits.max_depth = 1
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.results_per_query = 8
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.2

    engine = SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)
    tree = engine.run("search system")

    for child_id in tree.nodes[tree.root_id].children:
        child = tree.nodes[child_id]
        if child.status == "stopped":
            context = tree.expansions[child_id]
            assert context.stop_reason == "max_depth"
            assert any(event.event_type == "expand_stopped" for event in context.events)


def test_engine_applies_strict_as_of_cutoff_before_ranking() -> None:
    class TimedProvider:
        def search(self, query: str, max_results: int) -> list[SearchDocument]:
            docs = [
                SearchDocument(
                    title="alpha earnings preview",
                    snippet="alpha earnings preview",
                    url="https://news.example.com/alpha-preview",
                    source="news.example.com",
                    query=query,
                    rank=1,
                    published_at="2026-04-08T08:30:00Z",
                    published_at_precision="datetime",
                ),
                SearchDocument(
                    title="alpha intraday rumor",
                    snippet="alpha intraday rumor",
                    url="https://news.example.com/alpha-rumor",
                    source="news.example.com",
                    query=query,
                    rank=2,
                    published_at="2026-04-08T12:30:00Z",
                    published_at_precision="datetime",
                ),
                SearchDocument(
                    title="alpha dated article",
                    snippet="alpha dated article",
                    url="https://news.example.com/alpha-dated",
                    source="news.example.com",
                    query=query,
                    rank=3,
                    published_at="2026-04-08",
                    published_at_precision="date",
                ),
                SearchDocument(
                    title="alpha older filing",
                    snippet="alpha older filing",
                    url="https://sec.gov/alpha-filing",
                    source="sec.gov",
                    query=query,
                    rank=4,
                    published_at="2026-04-07T18:00:00Z",
                    published_at_precision="datetime",
                ),
            ]
            return docs[:max_results]

        def enrich_documents(self, docs: list[SearchDocument], max_docs: int) -> list[SearchDocument]:
            return [doc.clone() for doc in docs]

    config = EngineConfig()
    config.limits.max_depth = 1
    config.limits.min_results = 2
    config.limits.min_cluster_size = 2
    config.limits.results_per_query = 4
    config.similarity.dedupe_threshold = 0.999
    config.similarity.scope_threshold = 0.1
    config.retrieval.enable_page_fetch = False

    engine = SearchTreeEngine(provider=TimedProvider(), embedder=KeywordEmbedder(), config=config)
    tree = engine.run(
        SearchRequest(
            query="ALPHA stock earnings",
            max_depth=1,
            results_per_query=4,
            metadata={"as_of": "2026-04-08T09:00:00Z"},
        )
    )

    root = tree.nodes[tree.root_id]
    kept_urls = {doc.url for doc in root.docs}

    assert "https://news.example.com/alpha-preview" in kept_urls
    assert "https://sec.gov/alpha-filing" in kept_urls
    assert "https://news.example.com/alpha-rumor" not in kept_urls
    assert "https://news.example.com/alpha-dated" not in kept_urls
    assert all(doc.metadata.get("as_of") == "2026-04-08T09:00:00Z" for doc in root.docs)
    assert any(event.event_type == "as_of_cutoff" for event in tree.logs)
