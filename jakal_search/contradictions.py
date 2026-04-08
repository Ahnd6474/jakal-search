from __future__ import annotations

from collections import Counter

from .retrieval import passage_signature, tokenize_text
from .types import SearchDocument
from .utils import jaccard_similarity

NEGATION_HINTS = {"not", "no", "never", "false", "wrong", "myth", "debunk", "against", "deny", "denies"}
SUPPORT_HINTS = {"is", "are", "does", "supports", "shows", "confirms", "evidence", "study", "official"}


def annotate_contradictions(docs: list[SearchDocument]) -> dict[str, float | int]:
    if not docs:
        return {"cluster_count": 0, "conflicting_cluster_count": 0, "penalty": 0.0}

    clusters: list[list[SearchDocument]] = []
    signatures: list[set[str]] = []
    for doc in docs:
        text = doc.passages[0].text if doc.passages else doc.text
        doc.contradiction_label = infer_stance(text)
        token_set = set(tokenize_text(passage_signature(text) or text))
        assigned = False
        for index, cluster_tokens in enumerate(signatures):
            if jaccard_similarity(token_set, cluster_tokens) >= 0.4:
                clusters[index].append(doc)
                signatures[index] |= token_set
                doc.contradiction_cluster_id = index
                assigned = True
                break
        if not assigned:
            doc.contradiction_cluster_id = len(clusters)
            clusters.append([doc])
            signatures.append(set(token_set))

    conflicting_clusters = 0
    contradiction_strength = 0.0
    for cluster_id, cluster_docs in enumerate(clusters):
        labels = Counter(doc.contradiction_label for doc in cluster_docs)
        has_support = labels["support"] > 0
        has_contradiction = labels["contradict"] > 0
        if has_support and has_contradiction:
            conflicting_clusters += 1
            cluster_penalty = labels["contradict"] / max(len(cluster_docs), 1)
            contradiction_strength += cluster_penalty
            for doc in cluster_docs:
                if doc.contradiction_label == "contradict":
                    doc.metadata["contradiction_penalty"] = cluster_penalty
                else:
                    doc.metadata["contradiction_penalty"] = cluster_penalty * 0.35
        else:
            for doc in cluster_docs:
                doc.metadata["contradiction_penalty"] = 0.0
    return {
        "cluster_count": len(clusters),
        "conflicting_cluster_count": conflicting_clusters,
        "penalty": contradiction_strength / max(len(clusters), 1),
    }


def infer_stance(text: str) -> str:
    lowered = text.lower()
    contradiction_hits = sum(1 for token in NEGATION_HINTS if token in lowered)
    support_hits = sum(1 for token in SUPPORT_HINTS if token in lowered)
    if contradiction_hits > support_hits:
        return "contradict"
    if support_hits > contradiction_hits:
        return "support"
    return "neutral"
