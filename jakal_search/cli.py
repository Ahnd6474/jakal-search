from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import EngineConfig
from .engine import build_default_engine
from .output import render_output
from .types import SearchRequest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Recursive evidence search engine.")
    parser.add_argument("--version", action="version", version=f"jakal-search {__version__}")
    parser.add_argument("query", help="Root search query")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-topics", type=int, default=24)
    parser.add_argument("--max-nodes", dest="max_topics", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-width", type=int, default=6)
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument(
        "--no-page-fetch",
        action="store_true",
        help="Disable landing-page fetch and passage extraction.",
    )
    parser.add_argument(
        "--fetch-top-k",
        type=int,
        default=6,
        help="How many top provider results to enrich with page content.",
    )
    parser.add_argument(
        "--provider-pack",
        choices=["auto", "general", "technical", "research", "news", "market_news"],
        default="auto",
        help="Free provider pack selection strategy.",
    )
    parser.add_argument(
        "--stock-news",
        action="store_true",
        help="Shortcut for market-news collection settings.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Strict cutoff timestamp. Only documents published at or before this timestamp are kept.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable landing-page cache.",
    )
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
        "--expansion-model",
        type=Path,
        default=None,
        help="Path to a trained expansion decision head (.pt).",
    )
    parser.add_argument(
        "--branch-model",
        dest="expansion_model",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
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
        choices=["report", "urls", "json", "answer", "records"],
        default="report",
        help="Output format for stdout",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = EngineConfig()
    config.limits.max_depth = args.max_depth
    config.limits.max_total_topics = args.max_topics
    config.limits.frontier_width = args.frontier_width
    config.limits.results_per_query = args.results_per_query
    config.retrieval.enable_page_fetch = not args.no_page_fetch
    config.retrieval.fetch_top_k = args.fetch_top_k
    config.retrieval.provider_pack = "market_news" if args.stock_news else args.provider_pack
    config.retrieval.enable_cache = not args.no_cache
    if args.stock_news:
        config.retrieval.freshness_weight = max(config.retrieval.freshness_weight, 0.2)
        config.scoring.freshness_weight = max(config.scoring.freshness_weight, 0.8)
    config.transformer_device = args.device
    config.trust_model_path = None if args.trust_model is None else str(args.trust_model)
    config.expansion_model_path = None if args.expansion_model is None else str(args.expansion_model)
    config.topic_reranker_model_path = None if args.topic_reranker_model is None else str(args.topic_reranker_model)
    config.falsehood_model_path = None if args.claim_model is None else str(args.claim_model)

    engine = build_default_engine(config)
    request = SearchRequest(
        query=args.query,
        max_depth=args.max_depth,
        max_total_topics=args.max_topics,
        frontier_width=args.frontier_width,
        results_per_query=args.results_per_query,
        metadata={"as_of": args.as_of} if args.as_of else {},
    )
    run = engine.run(request)
    print(render_output(run, args.format))

    if args.json_out is not None:
        args.json_out.write_text(render_output(run, "json"), encoding="utf-8")
        print(f"\nWrote JSON snapshot to {args.json_out}")
    return 0
