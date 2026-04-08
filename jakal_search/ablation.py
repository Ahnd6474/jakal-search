from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .config import EngineConfig
from .engine import build_default_engine
from .evaluation import build_feedback_judgments, build_seed_judgments, evaluate_engine
from .tuning import clone_config


@dataclass(slots=True)
class AblationResult:
    name: str
    metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "metrics": self.metrics}


def build_ablation_configs(base_config: EngineConfig) -> list[tuple[str, EngineConfig]]:
    variants: list[tuple[str, EngineConfig]] = []
    variants.append(("full", clone_config(base_config)))

    no_fetch = clone_config(base_config)
    no_fetch.retrieval.enable_page_fetch = False
    variants.append(("no_page_fetch", no_fetch))

    no_entity = clone_config(base_config)
    no_entity.retrieval.entity_weight = 0.0
    no_entity.scoring.entity_weight = 0.0
    variants.append(("no_entity_boost", no_entity))

    no_freshness = clone_config(base_config)
    no_freshness.retrieval.freshness_weight = 0.0
    no_freshness.scoring.freshness_weight = 0.0
    variants.append(("no_freshness", no_freshness))

    no_falsehood = clone_config(base_config)
    no_falsehood.falsehood.block_high_risk = False
    no_falsehood.falsehood.penalty_weight = 0.0
    no_falsehood.scoring.falsehood_weight = 0.0
    variants.append(("no_falsehood_penalty", no_falsehood))

    no_contradiction = clone_config(base_config)
    no_contradiction.scoring.contradiction_weight = 0.0
    variants.append(("no_contradiction_penalty", no_contradiction))

    general_only = clone_config(base_config)
    general_only.retrieval.provider_pack = "general"
    variants.append(("general_pack_only", general_only))
    return variants


def run_ablation(base_config: EngineConfig, *, feedback_dir: Path | None = None) -> list[AblationResult]:
    judged_queries = build_seed_judgments()
    if feedback_dir is not None and feedback_dir.exists():
        judged_queries.extend(build_feedback_judgments(feedback_dir))

    results: list[AblationResult] = []
    for name, config in build_ablation_configs(base_config):
        metrics = evaluate_engine(build_default_engine, config, judged_queries)
        results.append(AblationResult(name=name, metrics=metrics.to_dict()))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline feature ablations for jakal-search.")
    parser.add_argument("--feedback-dir", type=Path, default=Path("outputs") / "feedback")
    parser.add_argument("--json-out", type=Path, default=Path("outputs") / "evaluation" / "ablations.json")
    args = parser.parse_args()

    results = run_ablation(EngineConfig(), feedback_dir=args.feedback_dir)
    payload = {"ablations": [result.to_dict() for result in results]}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
