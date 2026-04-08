from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import EngineConfig
from .engine import SearchTreeEngine, build_default_engine
from .tuning import BENCHMARK_CASES
from .types import SearchDocument, SearchTree


@dataclass(slots=True)
class JudgedDocument:
    url: str
    relevance: int


@dataclass(slots=True)
class JudgedQuery:
    query: str
    judgments: list[JudgedDocument]


@dataclass(slots=True)
class MetricBundle:
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    diversity_at_10: float
    query_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mrr_at_10": round(self.mrr_at_10, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "diversity_at_10": round(self.diversity_at_10, 4),
            "query_count": self.query_count,
        }


def load_judged_queries(path: Path) -> list[JudgedQuery]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judged_queries: list[JudgedQuery] = []
    for row in rows:
        judged_queries.append(
            JudgedQuery(
                query=str(row["query"]),
                judgments=[JudgedDocument(url=str(item["url"]), relevance=int(item["relevance"])) for item in row["judgments"]],
            )
        )
    return judged_queries


def build_seed_judgments() -> list[JudgedQuery]:
    judged_queries: list[JudgedQuery] = []
    for case in BENCHMARK_CASES:
        judgments: list[JudgedDocument] = []
        for group in case.expected_topic_groups:
            for keyword in group:
                url = f"https://example.org/{case.name}/{keyword}"
                judgments.append(JudgedDocument(url=url, relevance=2))
        for term in case.forbidden_terms:
            judgments.append(JudgedDocument(url=f"https://spam.example/{case.name}/{term}", relevance=0))
        judged_queries.append(JudgedQuery(query=case.query, judgments=judgments))
    return judged_queries


def build_feedback_judgments(storage_dir: Path) -> list[JudgedQuery]:
    runs_dir = storage_dir / "runs"
    if not runs_dir.exists():
        return []
    judged_queries: list[JudgedQuery] = []
    for run_path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        feedback = payload.get("feedback")
        if not feedback:
            continue
        winning_slot = str(feedback.get("choice") or "").upper()
        judgments: list[JudgedDocument] = []
        for option in payload.get("options", []):
            relevance = 1
            if option.get("slot") == winning_slot:
                relevance = 2
            for node in option.get("tree", {}).get("nodes", {}).values():
                for doc in node.get("docs", []):
                    judgments.append(JudgedDocument(url=str(doc["url"]), relevance=relevance))
        judged_queries.append(JudgedQuery(query=str(payload.get("query") or ""), judgments=judgments))
    return judged_queries


def evaluate_engine(
    engine_factory: Callable[[EngineConfig], SearchTreeEngine],
    config: EngineConfig,
    judged_queries: list[JudgedQuery],
) -> MetricBundle:
    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []
    recall_scores: list[float] = []
    diversity_scores: list[float] = []

    for judged_query in judged_queries:
        engine = engine_factory(config)
        tree = engine.run(judged_query.query)
        docs = unique_documents(tree)[:10]
        relevance_by_url = {judged.url: judged.relevance for judged in judged_query.judgments}
        gains = [relevance_by_url.get(doc.url, 0) for doc in docs]
        ndcg_scores.append(_ndcg(gains, sorted(relevance_by_url.values(), reverse=True)[:10]))
        mrr_scores.append(_mrr(gains))
        relevant_total = sum(1 for relevance in relevance_by_url.values() if relevance > 0)
        recall_scores.append(sum(1 for gain in gains if gain > 0) / max(relevant_total, 1))
        diversity_scores.append(len({doc.source for doc in docs}) / max(len(docs), 1))

    return MetricBundle(
        ndcg_at_10=sum(ndcg_scores) / max(len(ndcg_scores), 1),
        mrr_at_10=sum(mrr_scores) / max(len(mrr_scores), 1),
        recall_at_10=sum(recall_scores) / max(len(recall_scores), 1),
        diversity_at_10=sum(diversity_scores) / max(len(diversity_scores), 1),
        query_count=len(judged_queries),
    )


def unique_documents(tree: SearchTree) -> list[SearchDocument]:
    by_url: dict[str, SearchDocument] = {}
    for node in sorted(tree.nodes.values(), key=lambda item: (item.depth, -item.score, item.node_id)):
        for doc in node.docs:
            existing = by_url.get(doc.url)
            if existing is None or doc.retrieval_score > existing.retrieval_score:
                by_url[doc.url] = doc
    return sorted(by_url.values(), key=lambda doc: (-doc.retrieval_score, -doc.trust_score, doc.rank))


def _dcg(gains: list[int]) -> float:
    score = 0.0
    for index, gain in enumerate(gains, start=1):
        score += ((2**gain) - 1) / math.log2(index + 1)
    return score


def _ndcg(gains: list[int], ideal: list[int]) -> float:
    ideal_score = _dcg(ideal)
    if ideal_score <= 0:
        return 0.0
    return _dcg(gains) / ideal_score


def _mrr(gains: list[int]) -> float:
    for index, gain in enumerate(gains, start=1):
        if gain > 0:
            return 1.0 / index
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate jakal-search against judged queries.")
    parser.add_argument("--judgments", type=Path, default=None)
    parser.add_argument("--feedback-dir", type=Path, default=Path("outputs") / "feedback")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    judged_queries = build_seed_judgments()
    if args.feedback_dir.exists():
        judged_queries.extend(build_feedback_judgments(args.feedback_dir))
    if args.judgments is not None and args.judgments.exists():
        judged_queries.extend(load_judged_queries(args.judgments))

    config = EngineConfig()
    metrics = evaluate_engine(build_default_engine, config, judged_queries)
    payload = metrics.to_dict()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
