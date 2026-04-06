from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import EngineConfig
from .engine import build_default_engine
from .output import render_output
from .types import SearchRequest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Tree-based exploratory search engine.")
    parser.add_argument("query", help="Root search query")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--frontier-width", type=int, default=6)
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cpu, cuda, mps, xpu, or directml",
    )
    parser.add_argument(
        "--trust-model",
        type=Path,
        default=None,
        help="Path to a trained supervised trust head (.pt).",
    )
    parser.add_argument(
        "--branch-model",
        type=Path,
        default=None,
        help="Path to a trained branch decision head (.pt).",
    )
    parser.add_argument(
        "--topic-reranker-model",
        type=Path,
        default=None,
        help="Path to a trained topic/query reranker head (.pt).",
    )
    parser.add_argument(
        "--claim-model",
        type=Path,
        default=None,
        help="Path to a trained false-claim head (.pt).",
    )
    parser.add_argument(
        "--format",
        choices=["report", "urls", "tree", "json"],
        default="report",
        help="Output format for stdout",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = EngineConfig()
    config.limits.max_depth = args.max_depth
    config.limits.max_total_nodes = args.max_nodes
    config.limits.frontier_width = args.frontier_width
    config.limits.results_per_query = args.results_per_query
    config.transformer_device = args.device
    config.trust_model_path = None if args.trust_model is None else str(args.trust_model)
    config.branch_model_path = None if args.branch_model is None else str(args.branch_model)
    config.topic_reranker_model_path = None if args.topic_reranker_model is None else str(args.topic_reranker_model)
    config.falsehood_model_path = None if args.claim_model is None else str(args.claim_model)

    engine = build_default_engine(config)
    request = SearchRequest(
        query=args.query,
        max_depth=args.max_depth,
        max_total_nodes=args.max_nodes,
        frontier_width=args.frontier_width,
        results_per_query=args.results_per_query,
    )
    tree = engine.run(request)
    print(render_output(tree, args.format))

    if args.json_out is not None:
        args.json_out.write_text(render_output(tree, "json"), encoding="utf-8")
        print(f"\nWrote JSON tree to {args.json_out}")
    return 0
