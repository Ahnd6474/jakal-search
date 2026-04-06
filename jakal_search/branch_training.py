from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .scoring import BranchDecisionMLP, branch_metrics_to_vector
from .trust import SourceTrustScorer
from .tuning import BENCHMARK_CASES, SyntheticProvider, build_embedder, clone_config, default_tuning_config
from .types import BranchMetrics, SearchNode, SearchRequest, SearchTree, TopicMemoryItem
from .utils import mean_embedding


@dataclass(slots=True)
class BranchExample:
    metrics: BranchMetrics
    label: int
    source_case: str
    topic_name: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "metrics": asdict(self.metrics),
                "label": self.label,
                "source_case": self.source_case,
                "topic_name": self.topic_name,
            },
            ensure_ascii=False,
        )


@dataclass(slots=True)
class BranchTrainingMetrics:
    train_size: int
    val_size: int
    threshold: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float


def topic_docs(docs, keywords):
    return [doc for doc in docs if any(token in doc.text.lower() for token in keywords)]


def _make_engine(provider, embedder, config, query: str):
    from .engine import SearchTreeEngine

    engine = SearchTreeEngine(provider=provider, embedder=embedder, config=clone_config(config))
    query_vector = embedder.embed([query])[0]
    engine._root_query_vector = query_vector
    root = SearchNode(node_id="root", query=query, depth=0, centroid=query_vector, status="expanded")
    tree = SearchTree(root_id="root", request=SearchRequest(query=query))
    tree.add_node(root)
    return engine, tree, root


def _append_branch_example(
    examples: list[BranchExample],
    *,
    engine,
    tree: SearchTree,
    root: SearchNode,
    docs,
    label: int,
    source_case: str,
    topic_name: str,
) -> None:
    if len(docs) < 2:
        return
    centroid = mean_embedding(np.asarray([doc.embedding for doc in docs], dtype=np.float32))
    branch = engine._evaluate_branch(tree, root, docs, centroid)
    examples.append(BranchExample(metrics=branch.metrics, label=label, source_case=source_case, topic_name=topic_name))


def _clip_metric(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _synthetic_example(bucket: list[BranchExample], *, label: int, rng: np.random.Generator) -> BranchExample:
    anchor = bucket[int(rng.integers(0, len(bucket)))]
    peer = bucket[int(rng.integers(0, len(bucket)))]
    alpha = float(rng.uniform(0.25, 0.75))
    noise = rng.normal(0.0, 0.03 if label == 1 else 0.05, size=7)
    base = (alpha * branch_metrics_to_vector(anchor.metrics)) + ((1.0 - alpha) * branch_metrics_to_vector(peer.metrics))
    values = base + noise
    if label == 1:
        values[0] = _clip_metric(max(values[0], 0.32))
        values[1] = _clip_metric(max(values[1], 0.45))
        values[2] = _clip_metric(max(values[2], 0.28))
        values[4] = _clip_metric(max(values[4], 0.35))
        values[6] = _clip_metric(min(values[6], 0.45))
    else:
        values[6] = _clip_metric(max(values[6], 0.2))
    metrics = BranchMetrics(
        novelty=_clip_metric(values[0]),
        trust=_clip_metric(values[1]),
        scope=_clip_metric(values[2]),
        support=_clip_metric(values[3]),
        vector_consistency=_clip_metric(values[4]),
        size_score=_clip_metric(values[5]),
        drift=_clip_metric(values[6]),
    )
    return BranchExample(
        metrics=metrics,
        label=label,
        source_case=f"{anchor.source_case}:aug",
        topic_name=f"{anchor.topic_name}:aug",
    )


def _rebalance_examples(examples: list[BranchExample], *, target_count: int, seed: int) -> list[BranchExample]:
    rng = np.random.default_rng(seed)
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    per_label = max(target_count // 2, min(len(positives), len(negatives)))

    balanced: list[BranchExample] = []
    for label, bucket in ((1, positives), (0, negatives)):
        if len(bucket) >= per_label:
            indexes = rng.choice(len(bucket), size=per_label, replace=False)
            balanced.extend(bucket[int(index)] for index in indexes)
            continue
        balanced.extend(bucket)
        while len([item for item in balanced if item.label == label]) < per_label:
            balanced.append(_synthetic_example(bucket, label=label, rng=rng))

    rng.shuffle(balanced)
    return balanced


def build_branch_examples(
    *,
    embedder_kind: str,
    model_name: str,
    device: str,
    seed: int,
    target_count: int = 1200,
) -> list[BranchExample]:
    config = default_tuning_config()
    config.transformer_model = model_name
    config.transformer_device = device
    provider = SyntheticProvider()
    embedder = build_embedder(embedder_kind, model_name, device)
    scorer = SourceTrustScorer(config.trust, embedder)
    examples: list[BranchExample] = []
    prepared_docs_by_case = {}

    for case in BENCHMARK_CASES:
        docs = provider.search(case.query, config.limits.results_per_query)
        vectors = embedder.embed([doc.text for doc in docs])
        for doc, vector in zip(docs, vectors):
            doc.embedding = vector
        prepared_docs_by_case[case.name] = scorer.assess_documents(docs)

    for case in BENCHMARK_CASES:
        docs = [doc.clone() for doc in prepared_docs_by_case[case.name]]
        engine, tree, root = _make_engine(provider, embedder, config, case.query)

        positive_groups = []
        for expected_group in case.expected_topic_groups:
            group_docs = topic_docs(docs, expected_group)
            if len(group_docs) >= 2:
                positive_groups.append((expected_group[0], group_docs))

        for topic_name, group_docs in positive_groups:
            for subset_size in range(2, min(len(group_docs), 4) + 1):
                for subset in combinations(group_docs, subset_size):
                    _append_branch_example(
                        examples,
                        engine=engine,
                        tree=tree,
                        root=root,
                        docs=list(subset),
                        label=1,
                        source_case=case.name,
                        topic_name=topic_name,
                    )
            centroid = mean_embedding(np.asarray([doc.embedding for doc in group_docs], dtype=np.float32))
            engine._topic_memory.append(TopicMemoryItem(node_id=f"{case.name}:{topic_name}", embedding=centroid))

        forbidden_docs = [doc for doc in docs if any(term in doc.text.lower() for term in case.forbidden_terms)]
        for subset_size in range(2, min(len(forbidden_docs), 3) + 1):
            for subset in combinations(forbidden_docs, subset_size):
                _append_branch_example(
                    examples,
                    engine=engine,
                    tree=tree,
                    root=root,
                    docs=list(subset),
                    label=0,
                    source_case=case.name,
                    topic_name="forbidden",
                )

        if len(positive_groups) >= 2:
            first_group = positive_groups[0][1]
            second_group = positive_groups[1][1]
            for left_size in range(1, min(2, len(first_group)) + 1):
                for right_size in range(1, min(2, len(second_group)) + 1):
                    for left_subset in combinations(first_group, left_size):
                        for right_subset in combinations(second_group, right_size):
                            _append_branch_example(
                                examples,
                                engine=engine,
                                tree=tree,
                                root=root,
                                docs=[*left_subset, *right_subset],
                                label=0,
                                source_case=case.name,
                                topic_name="mixed",
                            )

        for topic_name, group_docs in positive_groups:
            for subset_size in range(2, min(len(group_docs), 4) + 1):
                for subset in combinations(group_docs, subset_size):
                    duplicate_docs = [doc.clone() for doc in subset]
                    duplicate = subset[0].clone()
                    duplicate.embedding = subset[0].embedding
                    duplicate.trust_score = subset[0].trust_score
                    duplicate_docs.append(duplicate)
                    _append_branch_example(
                        examples,
                        engine=engine,
                        tree=tree,
                        root=root,
                        docs=duplicate_docs,
                        label=0,
                        source_case=case.name,
                        topic_name=f"{topic_name}:duplicate",
                    )

        for other_case in BENCHMARK_CASES:
            if other_case.name == case.name:
                continue
            offscope_docs = [doc.clone() for doc in prepared_docs_by_case[other_case.name]]
            for subset_size in range(2, min(len(offscope_docs), 4) + 1):
                for subset in combinations(offscope_docs, subset_size):
                    _append_branch_example(
                        examples,
                        engine=engine,
                        tree=tree,
                        root=root,
                        docs=list(subset),
                        label=0,
                        source_case=case.name,
                        topic_name=f"offscope:{other_case.name}",
                    )
    return _rebalance_examples(examples, target_count=target_count, seed=seed)


def stratified_split(examples: list[BranchExample], *, val_ratio: float, seed: int) -> tuple[list[BranchExample], list[BranchExample]]:
    rng = random.Random(seed)
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_val = max(1, int(len(positives) * val_ratio))
    neg_val = max(1, int(len(negatives) * val_ratio))
    return positives[pos_val:] + negatives[neg_val:], positives[:pos_val] + negatives[:neg_val]


def train_branch_head(
    examples: list[BranchExample],
    output_dir: Path,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_ratio: float,
    seed: int,
) -> tuple[Path, BranchTrainingMetrics]:
    train_examples, val_examples = stratified_split(examples, val_ratio=val_ratio, seed=seed)
    train_x = np.asarray([branch_metrics_to_vector(example.metrics) for example in train_examples], dtype=np.float32)
    val_x = np.asarray([branch_metrics_to_vector(example.metrics) for example in val_examples], dtype=np.float32)
    train_y = np.asarray([example.label for example in train_examples], dtype=np.float32)
    val_y = np.asarray([example.label for example in val_examples], dtype=np.float32)

    torch.manual_seed(seed)
    model = BranchDecisionMLP(train_x.shape[1], hidden_dim)
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

    threshold, eval_metrics = evaluate_branch_model(model, val_x, val_y)
    metrics = BranchTrainingMetrics(
        train_size=len(train_examples),
        val_size=len(val_examples),
        threshold=threshold,
        val_accuracy=eval_metrics.val_accuracy,
        val_precision=eval_metrics.val_precision,
        val_recall=eval_metrics.val_recall,
        val_f1=eval_metrics.val_f1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "branch_head.pt"
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
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "threshold": threshold,
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path, metrics


def evaluate_branch_model(model: BranchDecisionMLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, BranchTrainingMetrics]:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(features.astype(np.float32)))).numpy()
    best_threshold = 0.5
    best_stats = (-1.0, 0.0, 0.0, 0.0)
    for threshold in np.linspace(0.3, 0.8, 51):
        predictions = (probs >= threshold).astype(np.float32)
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
    return best_threshold, BranchTrainingMetrics(
        train_size=0,
        val_size=len(labels),
        threshold=best_threshold,
        val_accuracy=float(accuracy),
        val_precision=float(precision),
        val_recall=float(recall),
        val_f1=float(f1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a branch decision head from the local benchmark suite.")
    parser.add_argument("--embedder", choices=["synthetic", "sentence-transformer"], default="synthetic")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "models")
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs") / "datasets" / "branch_dataset.jsonl")
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--target-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = build_branch_examples(
        embedder_kind=args.embedder,
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
        target_count=args.target_count,
    )
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_out.write_text("\n".join(example.to_json() for example in examples), encoding="utf-8")
    model_path, metrics = train_branch_head(
        examples,
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
                "count": len(examples),
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
