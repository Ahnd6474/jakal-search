from __future__ import annotations

import json
from collections import Counter
from typing import Literal

from .answering import compose_evidence_answer
from .types import SearchDocument, SearchNode, SearchTree
from .utils import age_in_seconds, parse_iso_datetime

OutputFormat = Literal["report", "urls", "tree", "json", "answer", "records"]


def render_output(tree: SearchTree, output_format: OutputFormat) -> str:
    if output_format == "report":
        return render_report(tree)
    if output_format == "urls":
        return render_urls(tree)
    if output_format == "tree":
        return render_tree(tree)
    if output_format == "json":
        return render_json(tree)
    if output_format == "answer":
        return render_answer(tree)
    if output_format == "records":
        return render_records(tree)
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
        contradiction_count = int(root.metrics.get("contradiction_clusters", 0))
        if contradiction_count > 0:
            lines.append(f"- Contradictory evidence clusters: {contradiction_count}.")
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
                f"support={_metric(node, 'support')}, "
                f"evidence={_metric(node, 'evidence_coverage')}, "
                f"diversity={_metric(node, 'source_diversity')}, "
                f"freshness={_metric(node, 'freshness')}, "
                f"entity={_metric(node, 'entity_alignment')}"
            )
            lines.append("   Documents:")
            for doc in _top_documents(node.docs):
                evidence = ""
                if doc.passages:
                    evidence = f" | evidence: {_trim(doc.passages[0].text, 110)}"
                lines.append(
                    f"   - {doc.title} ({doc.source}, retrieval={doc.retrieval_score:.3f}, trust={doc.trust_score:.3f}, fresh={doc.freshness_score:.3f}, stance={doc.contradiction_label}){evidence}"
                )

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


def render_answer(tree: SearchTree) -> str:
    return compose_evidence_answer(tree)


def render_records(tree: SearchTree) -> str:
    as_of = ""
    if tree.request is not None:
        as_of = str(tree.request.metadata.get("as_of") or "").strip()
    memberships = _topic_memberships(tree)
    records = [
        _document_record(
            doc,
            tree.nodes[tree.root_id].query,
            as_of=as_of or None,
            topic_memberships=memberships.get(doc.url, []),
        )
        for doc in _unique_documents(tree)
    ]
    return json.dumps(records, ensure_ascii=False, indent=2)


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
        key=lambda doc: (-doc.retrieval_score, -doc.trust_score, doc.rank, doc.source, doc.title.lower()),
    )


def _doc_priority(doc: SearchDocument) -> tuple[float, float, str]:
    return (doc.trust_score, -float(doc.rank), doc.title)


def _top_documents(documents: list[SearchDocument], limit: int = 3) -> list[SearchDocument]:
    return sorted(
        documents,
        key=lambda doc: (-doc.retrieval_score, -doc.trust_score, doc.rank, doc.source, doc.title.lower()),
    )[:limit]


def _metric(node: SearchNode, name: str) -> str:
    value = node.metrics.get(name)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _trim(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3].rstrip()}..."


def _document_record(
    doc: SearchDocument,
    query: str,
    *,
    as_of: str | None = None,
    topic_memberships: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    evidence = doc.passages[0].text if doc.passages else doc.content or doc.snippet
    embedding_text = " [SEP] ".join(
        part
        for part in [
            query.strip(),
            doc.title.strip(),
            doc.snippet.strip(),
            _trim(doc.content.strip(), 1600) if doc.content else "",
            evidence.strip(),
        ]
        if part
    )
    lag_seconds = age_in_seconds(doc.published_at, now=parse_iso_datetime(as_of)) if as_of else None
    return {
        "query": query,
        "as_of": as_of,
        "url": doc.url,
        "source": doc.source,
        "title": doc.title,
        "snippet": doc.snippet,
        "content": doc.content,
        "evidence": evidence,
        "published_at": doc.published_at,
        "published_at_precision": doc.published_at_precision,
        "lag_seconds": None if lag_seconds is None else round(lag_seconds, 3),
        "retrieval_score": round(doc.retrieval_score, 6),
        "dense_score": round(doc.dense_score, 6),
        "lexical_score": round(doc.lexical_score, 6),
        "provider_score": round(doc.provider_score, 6),
        "trust_score": round(doc.trust_score, 6),
        "freshness_score": round(doc.freshness_score, 6),
        "entity_score": round(doc.entity_score, 6),
        "claim_falsehood_score": round(doc.claim_falsehood_score, 6),
        "contradiction_label": doc.contradiction_label,
        "provider_names": list(doc.metadata.get("provider_names") or []),
        "query_intent": doc.metadata.get("query_intent"),
        "domain_pack": doc.metadata.get("domain_pack"),
        "entity_hints": list(doc.metadata.get("entity_hints") or []),
        "ticker_hints": list(doc.metadata.get("ticker_hints") or []),
        "theme_memberships": list(doc.metadata.get("theme_memberships") or []),
        "topic_memberships": list(topic_memberships or []),
        "embedding_text": embedding_text,
    }


def _topic_memberships(tree: SearchTree) -> dict[str, list[dict[str, object]]]:
    memberships: dict[str, list[dict[str, object]]] = {}
    for node in _topic_nodes(tree):
        if node.topic is None:
            continue
        membership = {
            "node_id": node.node_id,
            "label": node.topic.label,
            "query": node.topic.query,
            "score": round(node.score, 6),
            "keywords": list(node.topic.keywords),
            "candidate_scores": list(node.topic.candidate_scores),
        }
        for doc in node.docs:
            memberships.setdefault(doc.url, []).append(membership)
    return memberships
