from __future__ import annotations

from collections import Counter

from .types import SearchDocument, SearchRun


def compose_evidence_answer(run: SearchRun) -> str:
    root = run.topics[run.root_topic_id]
    docs = _unique_documents(run)
    top_docs = docs[:5]
    lines = [f"Query: {root.query}", "", "Evidence-First Answer"]
    if not top_docs:
        lines.append("- No high-confidence evidence was extracted.")
        return "\n".join(lines)

    for index, doc in enumerate(top_docs, start=1):
        evidence = doc.passages[0].text if doc.passages else doc.snippet or doc.title
        lines.append(
            f"{index}. {doc.title} [{doc.source}] "
            f"(retrieval={doc.retrieval_score:.3f}, trust={doc.trust_score:.3f}, freshness={doc.freshness_score:.3f})"
        )
        lines.append(f"   Evidence: {_trim(evidence, 180)}")

    contradiction_docs = [doc for doc in docs if doc.contradiction_label == "contradict"]
    if contradiction_docs:
        lines.extend(["", "Contradictions"])
        for doc in contradiction_docs[:3]:
            evidence = doc.passages[0].text if doc.passages else doc.snippet or doc.title
            lines.append(f"- {doc.title} [{doc.source}]")
            lines.append(f"  Evidence: {_trim(evidence, 160)}")

    source_counts = Counter(doc.source for doc in docs)
    lines.extend(["", "Source Coverage"])
    for source, count in source_counts.most_common(6):
        lines.append(f"- {source} ({count})")
    return "\n".join(lines)


def _unique_documents(run: SearchRun) -> list[SearchDocument]:
    by_url: dict[str, SearchDocument] = {}
    for topic in sorted(run.topics.values(), key=lambda item: (item.depth, -item.score, item.topic_id)):
        for doc in topic.docs:
            existing = by_url.get(doc.url)
            if existing is None or doc.retrieval_score > existing.retrieval_score:
                by_url[doc.url] = doc
    return sorted(by_url.values(), key=lambda doc: (-doc.retrieval_score, -doc.trust_score, doc.rank))


def _trim(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3].rstrip()}..."
