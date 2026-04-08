from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import numpy as np

from .reranking import TopicQueryMLP, topic_candidate_features
from .tuning import BENCHMARK_CASES, SyntheticProvider, build_embedder


@dataclass(slots=True)
class RerankerExample:
    candidate: str
    label: int
    source_case: str
    parent_query: str
    doc_indexes: tuple[int, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "candidate": self.candidate,
                "label": self.label,
                "source_case": self.source_case,
                "parent_query": self.parent_query,
                "doc_indexes": list(self.doc_indexes),
            },
            ensure_ascii=False,
        )


@dataclass(slots=True)
class RerankerMetrics:
    train_size: int
    val_size: int
    threshold: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float


def build_reranker_examples(
    *,
    embedder_kind: str,
    model_name: str,
    device: str,
    seed: int,
):
    rng = random.Random(seed)
    provider = SyntheticProvider()
    embedder = build_embedder(embedder_kind, model_name, device)
    dataset: list[tuple[np.ndarray, int, RerankerExample]] = []

    for case in BENCHMARK_CASES:
        docs = provider.search(case.query, 8)
        vectors = embedder.embed([doc.text for doc in docs])
        for doc, vector in zip(docs, vectors):
            doc.embedding = vector
            doc.trust_score = 0.7 if doc.source not in {"blogspot.com", "wordpress.com"} else 0.2

        positives = sorted({token for group in case.expected_topic_groups for token in group})
        negatives = sorted(set(case.forbidden_terms) | {"guide", "system", "official", "notes"})

        for token in positives:
            features = topic_candidate_features(parent_query=case.query, candidate_text=token, docs=docs, embedder=embedder)
            dataset.append((features, 1, RerankerExample(token, 1, case.name, case.query, tuple(range(min(3, len(docs)))))))
            combined = f"{case.query} {token}"
            features = topic_candidate_features(parent_query=case.query, candidate_text=combined, docs=docs, embedder=embedder)
            dataset.append((features, 1, RerankerExample(combined, 1, case.name, case.query, tuple(range(min(3, len(docs)))))))

        for token in negatives:
            features = topic_candidate_features(parent_query=case.query, candidate_text=token, docs=docs, embedder=embedder)
            dataset.append((features, 0, RerankerExample(token, 0, case.name, case.query, tuple(range(min(3, len(docs)))))))
            combined = f"{case.query} {token}"
            features = topic_candidate_features(parent_query=case.query, candidate_text=combined, docs=docs, embedder=embedder)
            dataset.append((features, 0, RerankerExample(combined, 0, case.name, case.query, tuple(range(min(3, len(docs)))))))

        distractor_case = rng.choice([item for item in BENCHMARK_CASES if item.name != case.name])
        distractor = distractor_case.expected_topic_groups[0][0]
        features = topic_candidate_features(parent_query=case.query, candidate_text=distractor, docs=docs, embedder=embedder)
        dataset.append((features, 0, RerankerExample(distractor, 0, case.name, case.query, tuple(range(min(3, len(docs)))))))
    return dataset


def stratified_split(rows, *, val_ratio: float, seed: int):
    rng = random.Random(seed)
    positives = [row for row in rows if row[1] == 1]
    negatives = [row for row in rows if row[1] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_val = max(1, int(len(positives) * val_ratio))
    neg_val = max(1, int(len(negatives) * val_ratio))
    return positives[pos_val:] + negatives[neg_val:], positives[:pos_val] + negatives[:neg_val]


def evaluate_model(model: TopicQueryMLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, RerankerMetrics]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(features.astype(np.float32)))).numpy()
    best_threshold = 0.5
    best_stats = (-1.0, 0.0, 0.0, 0.0)
    for threshold in np.linspace(0.3, 0.8, 51):
        predictions = (probabilities >= threshold).astype(np.float32)
        tp = float(np.sum((predictions == 1) & (labels == 1)))
        fp = float(np.sum((predictions == 1) & (labels == 0)))
        fn = float(np.sum((predictions == 0) & (labels == 1)))
        tn = float(np.sum((predictions == 0) & (labels == 0)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        candidate = (f1, accuracy, precision, recall)
        if candidate > best_stats:
            best_stats = candidate
            best_threshold = float(threshold)
    f1, accuracy, precision, recall = best_stats
    return best_threshold, RerankerMetrics(
        train_size=0,
        val_size=len(labels),
        threshold=best_threshold,
        val_accuracy=float(accuracy),
        val_precision=float(precision),
        val_recall=float(recall),
        val_f1=float(f1),
    )


def train_reranker(
    rows,
    output_dir: Path,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_ratio: float,
    seed: int,
) -> tuple[Path, RerankerMetrics]:
    train_rows, val_rows = stratified_split(rows, val_ratio=val_ratio, seed=seed)
    train_x = np.asarray([row[0] for row in train_rows], dtype=np.float32)
    val_x = np.asarray([row[0] for row in val_rows], dtype=np.float32)
    train_y = np.asarray([row[1] for row in train_rows], dtype=np.float32)
    val_y = np.asarray([row[1] for row in val_rows], dtype=np.float32)

    torch.manual_seed(seed)
    model = TopicQueryMLP(train_x.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight = torch.tensor([max(float(np.sum(train_y == 0.0)), 1.0) / max(float(np.sum(train_y == 1.0)), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    inputs = torch.from_numpy(train_x)
    targets = torch.from_numpy(train_y)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    threshold, metrics = evaluate_model(model, val_x, val_y)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "topic_reranker_head.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": hidden_dim,
            "threshold": threshold,
        },
        model_path,
    )
    model_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "threshold": threshold,
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the topic/query reranker on local benchmark candidates.")
    parser.add_argument("--embedder", choices=["synthetic", "sentence-transformer"], default="synthetic")
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "models")
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs") / "datasets" / "topic_reranker_dataset.jsonl")
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = build_reranker_examples(
        embedder_kind=args.embedder,
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
    )
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_out.write_text("\n".join(row[2].to_json() for row in rows), encoding="utf-8")
    model_path, metrics = train_reranker(
        rows,
        args.output_dir,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "dataset_path": str(args.dataset_out),
                "count": len(rows),
                "model_path": str(model_path),
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
