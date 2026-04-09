from __future__ import annotations

from jakal_search import RecursiveSearchEngine, SearchRun
from jakal_search.engine import SearchTreeEngine
from jakal_search.expansion_training import build_expansion_examples, train_expansion_head
from jakal_search.scoring import ExpansionDecisionModel
from jakal_search.types import ExpansionMetrics, SearchRequest, SearchTopic


def test_engine_aliases_resolve_to_runtime_types() -> None:
    assert SearchTreeEngine is RecursiveSearchEngine


def test_expansion_aliases_expose_runtime_types() -> None:
    metrics = ExpansionMetrics(
        novelty=0.8,
        trust=0.7,
        scope=0.9,
        support=0.6,
        vector_consistency=0.75,
        size_score=0.9,
        drift=0.1,
    )
    decision = ExpansionDecisionModel.__name__

    assert metrics.novelty == 0.8
    assert decision == "ExpansionDecisionModel"
    assert SearchRun.__name__ == "SearchRun"


def test_expansion_training_symbols_are_primary_api() -> None:
    assert build_expansion_examples.__name__ == "build_expansion_examples"
    assert train_expansion_head.__name__ == "train_expansion_head"


def test_topic_and_request_alias_properties_remain_available() -> None:
    topic = SearchTopic(topic_id="topic-1", query="alpha", depth=1)
    request = SearchRequest(query="alpha", max_total_topics=5)

    assert topic.node_id == "topic-1"
    assert request.max_total_nodes == 5
