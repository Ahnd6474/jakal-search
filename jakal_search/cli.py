from __future__ import annotations

import argparse
from pathlib import Path

from .config import EngineConfig
from .engine import build_default_engine
from .types import SearchTree


def main() -> int:
    parser = argparse.ArgumentParser(description="Tree-based exploratory search engine.")
    parser.add_argument("query", help="Root search query")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=24)
    parser.add_argument("--frontier-width", type=int, default=6)
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = EngineConfig()
    config.limits.max_depth = args.max_depth
    config.limits.max_total_nodes = args.max_nodes
    config.limits.frontier_width = args.frontier_width
    config.limits.results_per_query = args.results_per_query

    engine = build_default_engine(config)
    tree = engine.run(args.query)
    _print_tree(tree)

    if args.json_out is not None:
        args.json_out.write_text(engine.dumps(tree), encoding="utf-8")
        print(f"\nWrote JSON tree to {args.json_out}")
    return 0


def _print_tree(tree: SearchTree) -> None:
    ordered = sorted(tree.nodes.values(), key=lambda node: (node.depth, -node.score, node.node_id))
    for node in ordered:
        indent = "  " * node.depth
        label = f" [{node.cluster_label}]" if node.cluster_label else ""
        status = node.stop_reason or node.status
        print(f"{indent}- {node.node_id}: {node.query}{label} ({status}, score={node.score:.3f})")
        if node.metrics:
            metrics = ", ".join(f"{key}={value}" for key, value in node.metrics.items())
            print(f"{indent}  {metrics}")
