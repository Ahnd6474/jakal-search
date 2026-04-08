from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .policy import PairwiseSearchService


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="GUI bridge for pairwise jakal-search comparisons.")
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="Optional directory for policy state and feedback logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("compare", help="Generate two hidden-policy search results for the query.")
    subparsers.add_parser("feedback", help="Record which result the user preferred.")
    subparsers.add_parser("dashboard", help="Return developer-facing policy diagnostics.")
    args = parser.parse_args()

    try:
        payload = _read_payload()
        service = PairwiseSearchService(storage_dir=args.storage_dir)
        if args.command == "compare":
            result = service.compare(str(payload.get("query", "")))
        elif args.command == "feedback":
            result = service.record_feedback(
                str(payload.get("run_id", "")),
                str(payload.get("choice", "")),
            )
        else:
            result = service.dashboard()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
