from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import EngineConfig
from .embedding import SentenceTransformerEmbedder, TextEmbedder
from .engine import SearchTreeEngine
from .trust import SourceTrustScorer
from .types import SearchDocument, SearchRequest, SearchTree
from .utils import normalize_rows


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    query: str
    expected_topic_groups: tuple[tuple[str, ...], ...]
    forbidden_terms: tuple[str, ...] = ()
    preferred_sources: tuple[str, ...] = ()
    expected_children: int = 2
    expected_trusted_docs: int = 5


@dataclass(slots=True)
class CaseEvaluation:
    name: str
    score: float
    child_queries: list[str]
    kept_sources: list[str]
    trusted_count: int
    unique_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "child_queries": self.child_queries,
            "kept_sources": self.kept_sources,
            "trusted_count": self.trusted_count,
            "unique_count": self.unique_count,
        }


@dataclass(slots=True)
class TrustBenchmarkCase:
    name: str
    documents: tuple[SearchDocument, ...]
    expected_kept_titles: tuple[str, ...]
    expected_rejected_titles: tuple[str, ...]
    min_margin: float = 0.2


@dataclass(slots=True)
class TrustEvaluation:
    name: str
    score: float
    kept_titles: list[str]
    rejected_titles: list[str]
    trust_scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "kept_titles": self.kept_titles,
            "rejected_titles": self.rejected_titles,
            "trust_scores": {key: round(value, 4) for key, value in self.trust_scores.items()},
        }


@dataclass(slots=True)
class TuningResult:
    score: float
    config: EngineConfig
    evaluations: list[CaseEvaluation]
    trust_evaluations: list[TrustEvaluation]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "config": engine_config_to_dict(self.config),
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "trust_evaluations": [evaluation.to_dict() for evaluation in self.trust_evaluations],
        }


def make_doc(title: str, source: str, query: str, rank: int, snippet: str | None = None) -> SearchDocument:
    return SearchDocument(
        title=title,
        snippet=title if snippet is None else snippet,
        url=f"https://{source}/{title.replace(' ', '-')}",
        source=source,
        query=query,
        rank=rank,
    )


SEARCH_SYSTEM_DOCS = [
    make_doc("search policy governance roadmap", "arxiv.org", "search system", 1),
    make_doc("search policy standards", "acm.org", "search system", 2),
    make_doc("governance policy for search", "ieee.org", "search system", 3),
    make_doc("vector tracking for search branches", "github.com", "search system", 4),
    make_doc("tracking vector stability in search", "github.com", "search system", 5),
    make_doc("search branch vector drift", "openai.com", "search system", 6),
    make_doc("miracle secret search hack", "blogspot.com", "search system", 7),
    make_doc("conspiracy truth about search", "blogspot.com", "search system", 8),
]

AGENT_PLATFORM_DOCS = [
    make_doc("agent platform deployment workflow", "docs.python.org", "agent platform", 1),
    make_doc("container deployment for agent platforms", "github.com", "agent platform", 2),
    make_doc("agent rollout guide for platform teams", "openai.com", "agent platform", 3),
    make_doc("observability for agent platforms", "acm.org", "agent platform", 4),
    make_doc("platform tracing and monitoring", "github.com", "agent platform", 5),
    make_doc("latency dashboards for agents", "docs.python.org", "agent platform", 6),
    make_doc("secret platform growth hack", "blogspot.com", "agent platform", 7),
    make_doc("urgent giveaway for startup traction", "wordpress.com", "agent platform", 8),
]

OPEN_SOURCE_GOVERNANCE_DOCS = [
    make_doc("open source governance handbook", "acm.org", "open source governance", 1),
    make_doc("maintainer governance playbook", "github.com", "open source governance", 2),
    make_doc("community moderation checklist", "wordpress.com", "open source governance", 3),
    make_doc("maintainer onboarding notes", "medium.com", "open source governance", 4),
    make_doc("community support rotas", "substack.com", "open source governance", 5),
    make_doc("project policy for contributors", "docs.python.org", "open source governance", 6),
    make_doc("secret maintainer wealth formula", "blogspot.com", "open source governance", 7),
    make_doc("conspiracy about contributor rankings", "blogspot.com", "open source governance", 8),
]

MODEL_EVALUATION_DOCS = [
    make_doc("llm benchmark regression triage", "arxiv.org", "model evaluation", 1),
    make_doc("hallucination monitoring for model releases", "openai.com", "model evaluation", 2),
    make_doc("evaluation dataset drift checklist", "github.com", "model evaluation", 3),
    make_doc("safety eval incident postmortem", "acm.org", "model evaluation", 4),
    make_doc("release gate metrics for language models", "docs.python.org", "model evaluation", 5),
    make_doc("official benchmark loophole for instant wins", "wordpress.com", "model evaluation", 6),
    make_doc("guaranteed leaderboard domination hack", "blogspot.com", "model evaluation", 7),
    make_doc("secret eval bypass tutorial", "blogspot.com", "model evaluation", 8),
]

BENCHMARK_CASES = (
    BenchmarkCase(
        name="search-system",
        query="search system",
        expected_topic_groups=(("policy", "governance"), ("vector", "tracking", "drift")),
        forbidden_terms=("miracle", "conspiracy", "secret"),
        expected_trusted_docs=6,
    ),
    BenchmarkCase(
        name="agent-platform",
        query="agent platform",
        expected_topic_groups=(("deployment", "rollout", "container"), ("observability", "monitoring", "latency")),
        forbidden_terms=("secret", "giveaway", "urgent"),
        expected_trusted_docs=6,
    ),
    BenchmarkCase(
        name="open-source-governance",
        query="open source governance",
        expected_topic_groups=(("governance", "policy"), ("community", "maintainer", "moderation")),
        forbidden_terms=("secret", "conspiracy"),
        preferred_sources=("wordpress.com", "medium.com", "substack.com"),
        expected_trusted_docs=6,
    ),
    BenchmarkCase(
        name="model-evaluation",
        query="model evaluation",
        expected_topic_groups=(("benchmark", "evaluation", "regression"), ("hallucination", "safety", "monitoring")),
        forbidden_terms=("guaranteed", "hack", "bypass", "secret"),
        expected_trusted_docs=5,
    ),
)

TRUST_BENCHMARK_CASES = (
    TrustBenchmarkCase(
        name="adversarial-overlap",
        documents=(
            make_doc(
                "conspiracy detection benchmark dataset card",
                "arxiv.org",
                "trust benchmark",
                1,
                snippet="research note on classifying conspiracy narratives with reproducible evaluation",
            ),
            make_doc(
                "urgent security bulletin for package maintainers",
                "docs.python.org",
                "trust benchmark",
                2,
                snippet="official incident response guidance with concrete remediation steps",
            ),
            make_doc(
                "official documentation for guaranteed ranking wins",
                "blogspot.com",
                "trust benchmark",
                3,
                snippet="miracle workflow that secretly bypasses evaluation rules",
            ),
            make_doc(
                "academic style secret growth formula",
                "wordpress.com",
                "trust benchmark",
                4,
                snippet="guaranteed traffic and hidden tricks without evidence",
            ),
        ),
        expected_kept_titles=(
            "conspiracy detection benchmark dataset card",
            "urgent security bulletin for package maintainers",
        ),
        expected_rejected_titles=(
            "official documentation for guaranteed ranking wins",
            "academic style secret growth formula",
        ),
        min_margin=0.28,
    ),
    TrustBenchmarkCase(
        name="low-domain-high-signal",
        documents=(
            make_doc(
                "maintainer release checklist with rollback plan",
                "medium.com",
                "trust benchmark",
                1,
                snippet="postmortem-backed guide with commands, validation steps, and failure modes",
            ),
            make_doc(
                "community moderation runbook for issue triage",
                "substack.com",
                "trust benchmark",
                2,
                snippet="operator notes with examples, escalation paths, and tradeoffs",
            ),
            make_doc(
                "free maintainer wealth system",
                "blogspot.com",
                "trust benchmark",
                3,
                snippet="guaranteed contributor growth with secret formulas and no evidence",
            ),
            make_doc(
                "official sounding contribution miracle plan",
                "wordpress.com",
                "trust benchmark",
                4,
                snippet="instant results, hidden loopholes, and urgent action",
            ),
        ),
        expected_kept_titles=(
            "maintainer release checklist with rollback plan",
            "community moderation runbook for issue triage",
        ),
        expected_rejected_titles=(
            "free maintainer wealth system",
            "official sounding contribution miracle plan",
        ),
        min_margin=0.18,
    ),
)


class SyntheticProvider:
    def __init__(self) -> None:
        self._docs_by_query = {
            "search system": SEARCH_SYSTEM_DOCS,
            "agent platform": AGENT_PLATFORM_DOCS,
            "open source governance": OPEN_SOURCE_GOVERNANCE_DOCS,
            "model evaluation": MODEL_EVALUATION_DOCS,
        }

    def search(self, query: str, max_results: int) -> list[SearchDocument]:
        docs = self._docs_by_query.get(query.lower(), [])
        return [doc.clone() for doc in docs[:max_results]]


class KeywordBenchmarkEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(24, dtype=np.float32)
            keyword_groups = {
                0: ("search", "retrieval"),
                1: ("policy", "governance"),
                2: ("vector", "tracking", "drift"),
                3: ("deployment", "rollout", "container", "platform"),
                4: ("observability", "monitoring", "latency", "tracing", "dashboard"),
                5: ("community", "maintainer", "moderation", "contributors", "support"),
                6: ("secret", "miracle", "conspiracy", "giveaway", "urgent", "hack"),
            }
            for index, tokens in keyword_groups.items():
                for token in tokens:
                    if token in lowered:
                        vector[index] += 1.0
            for token in lowered.split():
                bucket = 7 + (sum(ord(char) for char in token) % 17)
                vector[bucket] += 0.12
            if not vector.any():
                vector[-1] = 1.0
            rows.append(vector)
        return normalize_rows(np.asarray(rows, dtype=np.float32))


class CachingEmbedder:
    def __init__(self, inner: TextEmbedder) -> None:
        self._inner = inner
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        missing = [text for text in texts if text not in self._cache]
        if missing:
            encoded = self._inner.embed(missing)
            for text, vector in zip(missing, encoded):
                self._cache[text] = np.asarray(vector, dtype=np.float32)

        return np.asarray([self._cache[text] for text in texts], dtype=np.float32)


def default_tuning_config() -> EngineConfig:
    config = EngineConfig(transformer_device="auto")
    config.limits.max_depth = 2
    config.limits.max_total_nodes = 10
    config.limits.frontier_width = 4
    config.limits.max_children_per_node = 2
    config.limits.min_results = 3
    config.limits.min_cluster_size = 3
    config.limits.results_per_query = 8
    config.retrieval.enable_page_fetch = False
    config.similarity.dedupe_threshold = 0.995
    config.similarity.novelty_threshold = 0.18
    config.similarity.scope_threshold = 0.2
    return config


def default_search_space() -> dict[tuple[str, str], list[float | int]]:
    return {
        ("limits", "results_per_query"): [8, 10],
        ("limits", "min_cluster_size"): [2, 3],
        ("similarity", "dedupe_threshold"): [0.992, 0.995],
        ("similarity", "novelty_threshold"): [0.16, 0.18],
        ("similarity", "scope_threshold"): [0.18, 0.2],
        ("trust", "min_domain_trust"): [0.22, 0.3],
        ("trust", "low_trust_semantic_threshold"): [0.74, 0.78],
        ("scoring", "min_continue_probability"): [0.46, 0.48],
        ("scoring", "min_vector_consistency"): [0.3, 0.32],
        ("scoring", "bias"): [-1.6, -1.45],
    }


def iter_candidate_configs(
    base_config: EngineConfig,
    search_space: dict[tuple[str, str], list[float | int]],
    limit: int | None = None,
) -> Iterable[EngineConfig]:
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    count = 0
    for combination in product(*values):
        candidate = clone_config(base_config)
        for (section, field_name), value in zip(keys, combination):
            setattr(getattr(candidate, section), field_name, value)
        yield candidate
        count += 1
        if limit is not None and count >= limit:
            return


def clone_config(config: EngineConfig) -> EngineConfig:
    return replace(
        config,
        limits=replace(config.limits),
        similarity=replace(config.similarity),
        trust=replace(
            config.trust,
            domain_weights=config.trust.domain_weights.copy(),
            blocked_domain_suffixes=tuple(config.trust.blocked_domain_suffixes),
            trusted_prototypes=tuple(config.trust.trusted_prototypes),
            suspicious_prototypes=tuple(config.trust.suspicious_prototypes),
        ),
        retrieval=replace(config.retrieval),
        scoring=replace(config.scoring),
        falsehood=replace(config.falsehood),
    )


def build_embedder(kind: str, model_name: str, device: str) -> TextEmbedder:
    if kind == "sentence-transformer":
        return CachingEmbedder(SentenceTransformerEmbedder(model_name, device=device))
    return KeywordBenchmarkEmbedder()


def evaluate_config(config: EngineConfig, embedder: TextEmbedder | None = None) -> TuningResult:
    provider = SyntheticProvider()
    embedder = embedder or KeywordBenchmarkEmbedder()
    evaluations: list[CaseEvaluation] = []
    trust_evaluations: list[TrustEvaluation] = []

    for case in BENCHMARK_CASES:
        engine = SearchTreeEngine(provider=provider, embedder=embedder, config=clone_config(config))
        request = SearchRequest(
            query=case.query,
            max_depth=config.limits.max_depth,
            max_total_nodes=config.limits.max_total_nodes,
            frontier_width=config.limits.frontier_width,
            results_per_query=config.limits.results_per_query,
            metadata={"benchmark_case": case.name},
        )
        tree = engine.run(request)
        evaluations.append(score_case(case, tree))

    for case in TRUST_BENCHMARK_CASES:
        trust_evaluations.append(score_trust_case(case, config, embedder))

    total_score = sum(item.score for item in evaluations) + sum(item.score for item in trust_evaluations)
    return TuningResult(
        score=total_score,
        config=clone_config(config),
        evaluations=evaluations,
        trust_evaluations=trust_evaluations,
    )


def score_case(case: BenchmarkCase, tree: SearchTree) -> CaseEvaluation:
    root = tree.nodes[tree.root_id]
    child_queries = [tree.nodes[child_id].query.lower() for child_id in root.children]
    kept_titles = [doc.title.lower() for doc in root.docs]
    kept_sources = [doc.source for doc in root.docs]
    child_scores = [tree.nodes[child_id].score for child_id in root.children]
    retrieval_scores = [doc.retrieval_score for doc in root.docs]
    evidence_scores = [doc.passages[0].score for doc in root.docs if doc.passages]

    score = 0.0
    if root.status == "expanded":
        score += 20.0

    child_delta = abs(len(root.children) - case.expected_children)
    score += max(0.0, 15.0 - (child_delta * 7.5))

    for group in case.expected_topic_groups:
        if any(any(keyword in query for keyword in group) for query in child_queries):
            score += 15.0

    if case.forbidden_terms and all(term not in " ".join(kept_titles) for term in case.forbidden_terms):
        score += 12.0

    if case.preferred_sources and any(source in case.preferred_sources for source in kept_sources):
        score += 8.0

    trusted_count = int(root.metrics.get("trusted_count", 0))
    unique_count = int(root.metrics.get("unique_count", 0))
    score += min(trusted_count / max(case.expected_trusted_docs, 1), 1.0) * 10.0
    score += min(unique_count / max(case.expected_trusted_docs, 1), 1.0) * 5.0
    if retrieval_scores:
        score += min(float(np.mean(retrieval_scores)), 1.0) * 6.0
    if evidence_scores:
        score += min(float(np.mean(evidence_scores)), 1.0) * 4.0

    if child_scores:
        score += float(np.mean(child_scores)) * 10.0

    return CaseEvaluation(
        name=case.name,
        score=score,
        child_queries=child_queries,
        kept_sources=kept_sources,
        trusted_count=trusted_count,
        unique_count=unique_count,
    )


def score_trust_case(case: TrustBenchmarkCase, config: EngineConfig, embedder: TextEmbedder) -> TrustEvaluation:
    scorer = SourceTrustScorer(config.trust, embedder)
    docs = [doc.clone() for doc in case.documents]
    vectors = embedder.embed([doc.text for doc in docs])
    for doc, vector in zip(docs, vectors):
        doc.embedding = vector

    kept_docs = scorer.assess_documents(docs)
    kept_titles = [doc.title for doc in kept_docs]
    rejected_titles = [doc.title for doc in docs if doc.title not in kept_titles]
    trust_scores = {doc.title: float(doc.trust_score) for doc in docs}

    score = 0.0
    expected_kept = set(case.expected_kept_titles)
    expected_rejected = set(case.expected_rejected_titles)
    actual_kept = set(kept_titles)
    actual_rejected = set(rejected_titles)

    score += 12.0 * len(expected_kept & actual_kept)
    score += 12.0 * len(expected_rejected & actual_rejected)

    if expected_kept == actual_kept:
        score += 10.0
    if expected_rejected == actual_rejected:
        score += 10.0

    if expected_kept and expected_rejected:
        min_kept = min(trust_scores[title] for title in expected_kept)
        max_rejected = max(trust_scores[title] for title in expected_rejected)
        margin = min_kept - max_rejected
        score += max(0.0, min(margin / max(case.min_margin, 1e-6), 1.0) * 10.0)

    return TrustEvaluation(
        name=case.name,
        score=score,
        kept_titles=kept_titles,
        rejected_titles=rejected_titles,
        trust_scores=trust_scores,
    )


def tune(
    limit: int | None = None,
    *,
    embedder_kind: str = "synthetic",
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
    device: str = "auto",
) -> tuple[TuningResult, list[TuningResult]]:
    base_config = default_tuning_config()
    base_config.transformer_model = model_name
    base_config.transformer_device = device
    embedder = build_embedder(embedder_kind, model_name, device)
    results = [
        evaluate_config(candidate, embedder=embedder)
        for candidate in iter_candidate_configs(base_config, default_search_space(), limit=limit)
    ]
    results.sort(
        key=lambda item: (
            item.score,
            item.config.scoring.min_continue_probability,
            item.config.similarity.scope_threshold,
            item.config.limits.results_per_query,
        ),
        reverse=True,
    )
    return results[0], results[:5]


def engine_config_to_dict(config: EngineConfig) -> dict[str, object]:
    return {
        "transformer_model": config.transformer_model,
        "transformer_device": config.transformer_device,
        "limits": asdict(config.limits),
        "similarity": asdict(config.similarity),
        "retrieval": asdict(config.retrieval),
        "trust": {
            "min_domain_trust": config.trust.min_domain_trust,
            "low_trust_semantic_threshold": config.trust.low_trust_semantic_threshold,
            "domain_score_weight": config.trust.domain_score_weight,
            "semantic_score_weight": config.trust.semantic_score_weight,
            "mlp_enabled_for_transformers": config.trust.mlp_enabled_for_transformers,
            "mlp_hidden_dim": config.trust.mlp_hidden_dim,
            "mlp_epochs": config.trust.mlp_epochs,
            "mlp_learning_rate": config.trust.mlp_learning_rate,
            "mlp_weight_decay": config.trust.mlp_weight_decay,
            "blocked_domain_suffixes": list(config.trust.blocked_domain_suffixes),
            "domain_weights": dict(config.trust.domain_weights),
            "trusted_prototypes": list(config.trust.trusted_prototypes),
            "suspicious_prototypes": list(config.trust.suspicious_prototypes),
        },
        "scoring": asdict(config.scoring),
        "falsehood": asdict(config.falsehood),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune heuristic search parameters against the local synthetic benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after evaluating this many candidates.")
    parser.add_argument(
        "--embedder",
        choices=["synthetic", "sentence-transformer"],
        default="synthetic",
        help="Which embedder to use during tuning.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-MiniLM-L3-v2",
        help="Sentence Transformer model name to use when --embedder sentence-transformer is selected.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cpu, cuda, mps, xpu, or directml.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("outputs") / "tuning" / "best_config.json",
        help="Where to write the best tuning result as JSON.",
    )
    args = parser.parse_args()

    best, top_results = tune(
        limit=args.limit,
        embedder_kind=args.embedder,
        model_name=args.model_name,
        device=args.device,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {
                "best": best.to_dict(),
                "top_results": [item.to_dict() for item in top_results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best score: {best.score:.4f}")
    print(json.dumps(best.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nWrote tuning results to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
