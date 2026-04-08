from __future__ import annotations

import random

import numpy as np

from jakal_search.config import EngineConfig
from jakal_search.engine import SearchTreeEngine
from jakal_search.policy import PairwiseSearchService
from jakal_search.types import SearchDocument
from jakal_search.utils import normalize_rows


def make_doc(title: str, source: str, query: str, rank: int = 1) -> SearchDocument:
    return SearchDocument(
        title=title,
        snippet=title,
        url=f"https://{source}/{title.replace(' ', '-')}",
        source=source,
        query=query,
        rank=rank,
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


def make_service(tmp_path):
    base_config = EngineConfig()
    base_config.limits.min_results = 3
    base_config.limits.min_cluster_size = 3
    base_config.similarity.dedupe_threshold = 0.995
    base_config.similarity.scope_threshold = 0.2

    def engine_factory(config: EngineConfig) -> SearchTreeEngine:
        return SearchTreeEngine(provider=FakeProvider(), embedder=KeywordEmbedder(), config=config)

    return PairwiseSearchService(
        storage_dir=tmp_path,
        base_config=base_config,
        engine_factory=engine_factory,
        rng=random.Random(7),
    )


def test_pairwise_compare_hides_internal_action_ids(tmp_path) -> None:
    service = make_service(tmp_path)

    result = service.compare("search system")

    assert result["run_id"].startswith("cmp-")
    assert len(result["options"]) == 2
    assert {option["slot"] for option in result["options"]} == {"A", "B"}
    assert all("action_id" not in option for option in result["options"])
    assert all(option["summary"]["document_count"] > 0 for option in result["options"])


def test_feedback_updates_dashboard_state(tmp_path) -> None:
    service = make_service(tmp_path)
    result = service.compare("search system")

    receipt = service.record_feedback(result["run_id"], "A")
    dashboard = service.dashboard()

    assert receipt["choice"] == "A"
    assert dashboard["feedback_count"] == 1
    assert dashboard["run_count"] == 1
    assert len(dashboard["leaderboard"]) >= 2
    assert sum(item["wins"] for item in dashboard["leaderboard"]) == 1
    assert sum(item["losses"] for item in dashboard["leaderboard"]) == 1


def test_feedback_cannot_be_submitted_twice(tmp_path) -> None:
    service = make_service(tmp_path)
    result = service.compare("search system")

    service.record_feedback(result["run_id"], "tie")

    try:
        service.record_feedback(result["run_id"], "B")
    except ValueError as exc:
        assert "already been submitted" in str(exc)
    else:
        raise AssertionError("expected duplicate feedback submission to fail")
