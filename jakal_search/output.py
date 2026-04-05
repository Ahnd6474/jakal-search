from __future__ import annotations

import json
from collections import Counter
from typing import Literal

from .types import SearchDocument, SearchNode, SearchTree

OutputFormat = Literal["report", "urls", "tree", "json"]


def render_output(tree: SearchTree, output_format: OutputFormat) -> str:
    if output_format == "report":
        return render_report(tree)
    if output_format == "urls":
        return render_urls(tree)
    if output_format == "tree":
        return render_tree(tree)
    if output_format == "json":
        return render_json(tree)
    raise ValueError(f"Unsupported output format: {output_format}")


def render_report(tree: SearchTree) -> str:
    root = tree.nodes[tree.root_id]
    topics = _topic_nodes(tree)
    documents = _unique_documents(tree)
    source_counts = Counter(doc.source for doc in documents)

    lines = [f"Query: {root.query}", "", "Summary"]
    if topics:
        topic_names = ", ".join(_topic_name(node) for node in topics[:3])
        top_sources = ", ".join(
            f"{source} ({count})" for source, count in source_counts.most_common(3)
        )
        lines.append(f"- Found {len(documents)} unique documents across {len(topics)} topic(s).")
        lines.append(f"- Top topics: {topic_names}.")
        if top_sources:
            lines.append(f"- Most common sources: {top_sources}.")
    else:
        lines.append("- No strong subtopics were extracted, so the result is presented as a single bundle.")
        lines.append(f"- Found {len(documents)} unique documents from the root query.")

    if topics:
        lines.extend(["", "Topics"])
        for index, node in enumerate(topics, start=1):
            lines.append(f"{index}. {_topic_name(node)}")
            lines.append(f"   Query: {node.query}")
            lines.append(f"   Status: {node.stop_reason or node.status}")
            lines.append(
                "   Signals: "
                f"score={node.score:.3f}, "
                f"trust={_metric(node, 'trust')}, "
                f"scope={_metric(node, 'scope')}, "
                f"support={_metric(node, 'support')}"
            )
            lines.append("   Documents:")
            for doc in _top_documents(node.docs):
                lines.append(f"   - {doc.title} ({doc.source})")

    if source_counts:
        lines.extend(["", "Sources"])
        for source, count in source_counts.most_common(8):
            lines.append(f"- {source} ({count})")

    return "\n".join(lines)


def render_urls(tree: SearchTree) -> str:
    return "\n".join(doc.url for doc in _unique_documents(tree))


def render_tree(tree: SearchTree) -> str:
    lines: list[str] = []
    ordered = sorted(tree.nodes.values(), key=lambda node: (node.depth, -node.score, node.node_id))
    for node in ordered:
        indent = "  " * node.depth
        label = f" [{node.cluster_label}]" if node.cluster_label else ""
        status = node.stop_reason or node.status
        lines.append(f"{indent}- {node.node_id}: {node.query}{label} ({status}, score={node.score:.3f})")
        if node.metrics:
            metrics = ", ".join(f"{key}={value}" for key, value in node.metrics.items())
            lines.append(f"{indent}  {metrics}")
    return "\n".join(lines)


def render_json(tree: SearchTree) -> str:
    return json.dumps(tree.to_dict(), ensure_ascii=False, indent=2)


def _topic_nodes(tree: SearchTree) -> list[SearchNode]:
    nodes = [node for node in tree.nodes.values() if node.node_id != tree.root_id and node.docs]
    return sorted(nodes, key=lambda node: (-node.score, node.depth, node.node_id))


def _topic_name(node: SearchNode) -> str:
    return node.cluster_label or node.query


def _unique_documents(tree: SearchTree) -> list[SearchDocument]:
    by_url: dict[str, SearchDocument] = {}
    ordered_nodes = sorted(tree.nodes.values(), key=lambda node: (node.depth, -node.score, node.node_id))
    for node in ordered_nodes:
        for doc in node.docs:
            existing = by_url.get(doc.url)
            if existing is None or _doc_priority(doc) > _doc_priority(existing):
                by_url[doc.url] = doc
    return sorted(
        by_url.values(),
        key=lambda doc: (-doc.trust_score, doc.rank, doc.source, doc.title.lower()),
    )


def _doc_priority(doc: SearchDocument) -> tuple[float, float, str]:
    return (doc.trust_score, -float(doc.rank), doc.title)


def _top_documents(documents: list[SearchDocument], limit: int = 3) -> list[SearchDocument]:
    return sorted(
        documents,
        key=lambda doc: (-doc.trust_score, doc.rank, doc.source, doc.title.lower()),
    )[:limit]


def _metric(node: SearchNode, name: str) -> str:
    value = node.metrics.get(name)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
