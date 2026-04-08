from __future__ import annotations

import copy
import json
import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from .config import EngineConfig
from .engine import SearchTreeEngine, build_default_engine
from .output import render_output
from .types import SearchDocument, SearchRequest, SearchTree

PreferenceChoice = Literal["A", "B", "tie"]


@dataclass(slots=True, frozen=True)
class HiddenSearchAction:
    action_id: str
    title: str
    repeat_count: int = 1
    limit_overrides: dict[str, int] = field(default_factory=dict)
    similarity_overrides: dict[str, float] = field(default_factory=dict)
    trust_overrides: dict[str, float] = field(default_factory=dict)
    scoring_overrides: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ActionStats:
    comparisons: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    rating: float = 1200.0
    total_quality: float = 0.0
    quality_observations: int = 0
    last_selected_at: str | None = None
    last_feedback_at: str | None = None

    @property
    def avg_quality(self) -> float:
        if self.quality_observations == 0:
            return 0.0
        return self.total_quality / self.quality_observations

    @property
    def win_rate(self) -> float:
        decisive = self.wins + self.losses
        if decisive == 0:
            return 0.0
        return self.wins / decisive

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": self.comparisons,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "rating": round(self.rating, 3),
            "avg_quality": round(self.avg_quality, 4),
            "quality_observations": self.quality_observations,
            "win_rate": round(self.win_rate, 4),
            "last_selected_at": self.last_selected_at,
            "last_feedback_at": self.last_feedback_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionStats:
        return cls(
            comparisons=int(payload.get("comparisons", 0)),
            wins=int(payload.get("wins", 0)),
            losses=int(payload.get("losses", 0)),
            ties=int(payload.get("ties", 0)),
            rating=float(payload.get("rating", 1200.0)),
            total_quality=float(payload.get("total_quality", 0.0)),
            quality_observations=int(payload.get("quality_observations", 0)),
            last_selected_at=payload.get("last_selected_at"),
            last_feedback_at=payload.get("last_feedback_at"),
        )


@dataclass(slots=True)
class PolicyState:
    version: int = 1
    feedback_count: int = 0
    run_count: int = 0
    actions: dict[str, ActionStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feedback_count": self.feedback_count,
            "run_count": self.run_count,
            "actions": {
                action_id: {
                    **stats.to_dict(),
                    "total_quality": round(stats.total_quality, 6),
                }
                for action_id, stats in self.actions.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyState:
        return cls(
            version=int(payload.get("version", 1)),
            feedback_count=int(payload.get("feedback_count", 0)),
            run_count=int(payload.get("run_count", 0)),
            actions={
                action_id: ActionStats.from_dict(stats)
                for action_id, stats in dict(payload.get("actions", {})).items()
            },
        )


DEFAULT_ACTIONS: tuple[HiddenSearchAction, ...] = (
    HiddenSearchAction(
        action_id="balanced",
        title="Balanced explorer",
        repeat_count=1,
        limit_overrides={
            "max_depth": 3,
            "max_total_nodes": 24,
            "frontier_width": 6,
            "results_per_query": 12,
            "max_children_per_node": 4,
        },
        scoring_overrides={"min_continue_probability": 0.48},
    ),
    HiddenSearchAction(
        action_id="deep_dive",
        title="Deep dive",
        repeat_count=2,
        limit_overrides={
            "max_depth": 4,
            "max_total_nodes": 32,
            "frontier_width": 5,
            "results_per_query": 10,
            "max_children_per_node": 3,
        },
        similarity_overrides={"scope_threshold": 0.27},
        scoring_overrides={"min_continue_probability": 0.44, "min_vector_consistency": 0.29},
    ),
    HiddenSearchAction(
        action_id="wide_scan",
        title="Wide scan",
        repeat_count=2,
        limit_overrides={
            "max_depth": 2,
            "max_total_nodes": 22,
            "frontier_width": 8,
            "results_per_query": 16,
            "max_children_per_node": 5,
            "min_cluster_size": 2,
        },
        similarity_overrides={"scope_threshold": 0.24, "novelty_threshold": 0.16},
    ),
    HiddenSearchAction(
        action_id="trust_first",
        title="Trust first",
        repeat_count=1,
        limit_overrides={
            "max_depth": 3,
            "max_total_nodes": 20,
            "frontier_width": 5,
            "results_per_query": 10,
            "max_children_per_node": 3,
        },
        trust_overrides={"min_domain_trust": 0.34, "low_trust_semantic_threshold": 0.82},
        scoring_overrides={"min_continue_probability": 0.5},
    ),
    HiddenSearchAction(
        action_id="novelty_hunter",
        title="Novelty hunter",
        repeat_count=3,
        limit_overrides={
            "max_depth": 3,
            "max_total_nodes": 28,
            "frontier_width": 7,
            "results_per_query": 14,
            "max_children_per_node": 4,
        },
        similarity_overrides={"dedupe_threshold": 0.9, "novelty_threshold": 0.12, "scope_threshold": 0.21},
        scoring_overrides={"min_continue_probability": 0.43},
    ),
    HiddenSearchAction(
        action_id="precision",
        title="Precision brancher",
        repeat_count=1,
        limit_overrides={
            "max_depth": 3,
            "max_total_nodes": 18,
            "frontier_width": 4,
            "results_per_query": 9,
            "max_children_per_node": 2,
        },
        similarity_overrides={"scope_threshold": 0.4},
        trust_overrides={"min_domain_trust": 0.28},
        scoring_overrides={"min_continue_probability": 0.54, "min_vector_consistency": 0.35},
    ),
)


class PairwiseSearchService:
    def __init__(
        self,
        *,
        storage_dir: Path | None = None,
        base_config: EngineConfig | None = None,
        engine_factory: Callable[[EngineConfig], SearchTreeEngine] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.storage_dir = Path("outputs") / "feedback" if storage_dir is None else storage_dir
        self.runs_dir = self.storage_dir / "runs"
        self.events_path = self.storage_dir / "feedback_events.jsonl"
        self.state_path = self.storage_dir / "policy_state.json"
        self.base_config = copy.deepcopy(base_config or EngineConfig())
        self.engine_factory = engine_factory or build_default_engine
        self.rng = rng or random.Random()
        self.actions = {action.action_id: action for action in DEFAULT_ACTIONS}
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def compare(self, query: str) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query is required")

        state = self._load_state()
        action_a, action_b = self._select_action_pair(state)
        created_at = _utc_now()
        run_id = f"cmp-{uuid.uuid4().hex[:12]}"

        option_records = []
        public_options = []
        for slot, action in zip(("A", "B"), (action_a, action_b)):
            result = self._run_action(normalized_query, action)
            option_records.append(
                {
                    "slot": slot,
                    "action_id": action.action_id,
                    "action_title": action.title,
                    "repeat_count": action.repeat_count,
                    **result,
                }
            )
            public_options.append(
                {
                    "slot": slot,
                    "summary": result["summary"],
                    "report": result["report"],
                    "tree": result["tree"],
                }
            )

        run_record = {
            "run_id": run_id,
            "query": normalized_query,
            "created_at": created_at,
            "feedback": None,
            "options": option_records,
        }
        self._write_run(run_id, run_record)

        state.run_count += 1
        for action_id in (action_a.action_id, action_b.action_id):
            state.actions[action_id].last_selected_at = created_at
        self._save_state(state)

        return {
            "run_id": run_id,
            "query": normalized_query,
            "created_at": created_at,
            "options": public_options,
            "policy": {
                "feedback_count": state.feedback_count,
                "run_count": state.run_count,
                "status_text": "엔진이 이전 비교 선택을 바탕으로 다음 후보를 자동 조정합니다.",
            },
        }

    def record_feedback(self, run_id: str, choice: PreferenceChoice) -> dict[str, Any]:
        normalized_choice = self._normalize_choice(choice)
        run_record = self._read_run(run_id)
        if run_record.get("feedback") is not None:
            raise ValueError("feedback has already been submitted for this run")

        option_a, option_b = run_record["options"]
        action_a = option_a["action_id"]
        action_b = option_b["action_id"]

        state = self._load_state()
        self._observe_quality(state.actions[action_a], float(option_a["quality_score"]))
        self._observe_quality(state.actions[action_b], float(option_b["quality_score"]))
        self._apply_preference(state, action_a, action_b, normalized_choice)
        state.feedback_count += 1
        feedback_at = _utc_now()
        state.actions[action_a].last_feedback_at = feedback_at
        state.actions[action_b].last_feedback_at = feedback_at
        self._save_state(state)

        event = {
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "query": run_record["query"],
            "submitted_at": feedback_at,
            "choice": normalized_choice,
            "options": [
                {
                    "slot": option_a["slot"],
                    "action_id": action_a,
                    "quality_score": option_a["quality_score"],
                },
                {
                    "slot": option_b["slot"],
                    "action_id": action_b,
                    "quality_score": option_b["quality_score"],
                },
            ],
        }
        self._append_event(event)

        run_record["feedback"] = {
            "choice": normalized_choice,
            "submitted_at": feedback_at,
        }
        self._write_run(run_id, run_record)

        return {
            "run_id": run_id,
            "choice": normalized_choice,
            "feedback_count": state.feedback_count,
            "message": "선호도가 저장됐고 다음 비교에 반영됩니다.",
        }

    def dashboard(self) -> dict[str, Any]:
        state = self._load_state()
        recent_events = self._read_recent_events(limit=10)
        leaderboard = []
        for action in DEFAULT_ACTIONS:
            stats = state.actions[action.action_id]
            leaderboard.append(
                {
                    "action_id": action.action_id,
                    "title": action.title,
                    "repeat_count": action.repeat_count,
                    **stats.to_dict(),
                }
            )
        leaderboard.sort(
            key=lambda item: (item["rating"], item["avg_quality"], -item["comparisons"]),
            reverse=True,
        )
        return {
            "feedback_count": state.feedback_count,
            "run_count": state.run_count,
            "leaderboard": leaderboard,
            "recent_events": recent_events,
        }

    def _run_action(self, query: str, action: HiddenSearchAction) -> dict[str, Any]:
        attempts = []
        for _ in range(action.repeat_count):
            config = self._config_for_action(action)
            engine = self.engine_factory(config)
            tree = engine.run(SearchRequest(query=query))
            attempts.append(
                {
                    "tree": tree,
                    "summary": summarize_tree(tree),
                    "report": render_output(tree, "report"),
                }
            )

        scored_attempts = [
            {
                **attempt,
                "quality_score": round(score_tree_quality(attempt["summary"]), 4),
            }
            for attempt in attempts
        ]
        best_attempt = max(scored_attempts, key=lambda item: item["quality_score"])
        return {
            "tree": best_attempt["tree"].to_dict(),
            "summary": {
                **best_attempt["summary"],
                "quality_score": best_attempt["quality_score"],
            },
            "report": best_attempt["report"],
            "quality_score": best_attempt["quality_score"],
            "attempt_count": action.repeat_count,
            "attempt_quality_scores": [item["quality_score"] for item in scored_attempts],
        }

    def _config_for_action(self, action: HiddenSearchAction) -> EngineConfig:
        config = copy.deepcopy(self.base_config)
        for field_name, value in action.limit_overrides.items():
            setattr(config.limits, field_name, value)
        for field_name, value in action.similarity_overrides.items():
            setattr(config.similarity, field_name, value)
        for field_name, value in action.trust_overrides.items():
            setattr(config.trust, field_name, value)
        for field_name, value in action.scoring_overrides.items():
            setattr(config.scoring, field_name, value)
        return config

    def _load_state(self) -> PolicyState:
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            state = PolicyState.from_dict(payload)
        else:
            state = PolicyState()
        for action in DEFAULT_ACTIONS:
            state.actions.setdefault(action.action_id, ActionStats())
        return state

    def _save_state(self, state: PolicyState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _select_action_pair(self, state: PolicyState) -> tuple[HiddenSearchAction, HiddenSearchAction]:
        scored_actions = []
        for action in DEFAULT_ACTIONS:
            stats = state.actions[action.action_id]
            sampled_rank = self._sample_rank(stats)
            scored_actions.append((action, sampled_rank))
        scored_actions.sort(key=lambda item: item[1], reverse=True)
        primary_action = scored_actions[0][0]
        primary_stats = state.actions[primary_action.action_id]

        remaining = [item[0] for item in scored_actions[1:]]
        secondary_action = max(
            remaining,
            key=lambda action: self._challenger_score(
                primary_stats=primary_stats,
                challenger_stats=state.actions[action.action_id],
            ),
        )
        return primary_action, secondary_action

    def _sample_rank(self, stats: ActionStats) -> float:
        exploitation = self.rng.betavariate(stats.wins + 1.0, stats.losses + 1.0)
        rating_signal = 1.0 / (1.0 + math.exp(-(stats.rating - 1200.0) / 220.0))
        quality_signal = min(stats.avg_quality, 1.0)
        exploration_bonus = 1.0 / math.sqrt(stats.comparisons + 1.0)
        return (exploitation * 0.5) + (rating_signal * 0.2) + (quality_signal * 0.1) + (exploration_bonus * 0.2)

    def _challenger_score(self, *, primary_stats: ActionStats, challenger_stats: ActionStats) -> float:
        closeness = 1.0 / (1.0 + abs(primary_stats.rating - challenger_stats.rating) / 160.0)
        uncertainty = 1.0 / math.sqrt(challenger_stats.comparisons + 1.0)
        quality_signal = min(challenger_stats.avg_quality, 1.0)
        return (closeness * 0.45) + (uncertainty * 0.45) + (quality_signal * 0.1)

    def _apply_preference(
        self,
        state: PolicyState,
        action_a: str,
        action_b: str,
        choice: PreferenceChoice,
    ) -> None:
        stats_a = state.actions[action_a]
        stats_b = state.actions[action_b]
        stats_a.comparisons += 1
        stats_b.comparisons += 1

        if choice == "A":
            score_a = 1.0
            score_b = 0.0
            stats_a.wins += 1
            stats_b.losses += 1
        elif choice == "B":
            score_a = 0.0
            score_b = 1.0
            stats_a.losses += 1
            stats_b.wins += 1
        else:
            score_a = 0.5
            score_b = 0.5
            stats_a.ties += 1
            stats_b.ties += 1

        expected_a = _expected_score(stats_a.rating, stats_b.rating)
        expected_b = _expected_score(stats_b.rating, stats_a.rating)
        k_factor = 24.0
        stats_a.rating += k_factor * (score_a - expected_a)
        stats_b.rating += k_factor * (score_b - expected_b)

    def _observe_quality(self, stats: ActionStats, quality_score: float) -> None:
        stats.total_quality += quality_score
        stats.quality_observations += 1

    def _append_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False))
            handle.write("\n")

    def _read_recent_events(self, *, limit: int) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

    def _write_run(self, run_id: str, payload: dict[str, Any]) -> None:
        run_path = self.runs_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_run(self, run_id: str) -> dict[str, Any]:
        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            raise ValueError(f"unknown run_id: {run_id}")
        return json.loads(run_path.read_text(encoding="utf-8"))

    def _normalize_choice(self, choice: str) -> PreferenceChoice:
        lowered = choice.strip().lower()
        if lowered in {"a", "left"}:
            return "A"
        if lowered in {"b", "right"}:
            return "B"
        if lowered in {"tie", "draw", "same"}:
            return "tie"
        raise ValueError("choice must be A, B, or tie")


def summarize_tree(tree: SearchTree) -> dict[str, Any]:
    unique_docs = unique_documents(tree)
    topic_nodes = [node for node in tree.nodes.values() if node.node_id != tree.root_id and node.docs]
    source_counts = Counter(doc.source for doc in unique_docs)
    avg_trust = 0.0 if not unique_docs else sum(doc.trust_score for doc in unique_docs) / len(unique_docs)
    expanded_nodes = [node for node in tree.nodes.values() if node.status == "expanded"]
    return {
        "topic_count": len(topic_nodes),
        "document_count": len(unique_docs),
        "avg_trust_score": round(avg_trust, 4),
        "expanded_node_count": len(expanded_nodes),
        "top_sources": [
            {"source": source, "count": count}
            for source, count in source_counts.most_common(5)
        ],
        "top_topics": [
            {
                "label": node.cluster_label or node.query,
                "query": node.query,
                "score": round(node.score, 4),
                "trust": round(float(node.metrics.get("trust", 0.0)), 4),
                "support": round(float(node.metrics.get("support", 0.0)), 4),
            }
            for node in sorted(topic_nodes, key=lambda item: item.score, reverse=True)[:5]
        ],
        "top_documents": [
            {
                "title": doc.title,
                "source": doc.source,
                "url": doc.url,
                "snippet": doc.snippet,
                "trust_score": round(doc.trust_score, 4),
            }
            for doc in unique_docs[:6]
        ],
    }


def score_tree_quality(summary: dict[str, Any]) -> float:
    topic_score = min(float(summary["topic_count"]) / 4.0, 1.0)
    document_score = min(float(summary["document_count"]) / 14.0, 1.0)
    trust_score = min(float(summary["avg_trust_score"]), 1.0)
    expansion_score = min(float(summary["expanded_node_count"]) / 5.0, 1.0)
    return (topic_score * 0.2) + (document_score * 0.3) + (trust_score * 0.35) + (expansion_score * 0.15)


def unique_documents(tree: SearchTree) -> list[SearchDocument]:
    by_url: dict[str, SearchDocument] = {}
    ordered_nodes = sorted(tree.nodes.values(), key=lambda node: (node.depth, -node.score, node.node_id))
    for node in ordered_nodes:
        for doc in node.docs:
            existing = by_url.get(doc.url)
            if existing is None or _document_priority(doc) > _document_priority(existing):
                by_url[doc.url] = doc
    return sorted(
        by_url.values(),
        key=lambda doc: (-doc.trust_score, doc.rank, doc.source, doc.title.lower()),
    )


def _document_priority(doc: SearchDocument) -> tuple[float, float, str]:
    return (doc.trust_score, -float(doc.rank), doc.title)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
