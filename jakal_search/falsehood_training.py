from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import torch
from bs4 import BeautifulSoup
from torch import nn

from .embedding import SentenceTransformerEmbedder
from .falsehood import ClaimFalsehoodMLP

NEWS_RSS_FEEDS = (
    ("bbc_world_negative", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("bbc_technology_negative", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("npr_news_negative", "https://www.npr.org/rss/rss.php?id=1001"),
)
SNOPES_FALSE_URLS = (
    ("snopes_false_positive", "https://www.snopes.com/fact-check/rating/false/"),
    ("snopes_mostly_false_positive", "https://www.snopes.com/fact-check/rating/mostly-false/"),
    ("snopes_true_negative", "https://www.snopes.com/fact-check/rating/true/"),
)
POLITIFACT_RULINGS = (
    ("politifact_false_positive", "false", 1),
    ("politifact_mostly_false_positive", "mostly-false", 1),
    ("politifact_pants_fire_positive", "pants-fire", 1),
    ("politifact_true_negative", "true", 0),
    ("politifact_mostly_true_negative", "mostly-true", 0),
)


@dataclass(slots=True)
class ClaimDatasetExample:
    text: str
    label: int
    source: str
    source_url: str
    item_id: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(slots=True)
class TrainingMetrics:
    train_size: int
    val_size: int
    threshold: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float


def normalize_text(*parts: str) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    return " ".join(text.split())


def strip_verdict_markers(text: str) -> str:
    cleaned = re.sub(
        r"\b(false|mostly false|pants on fire|pants-fire|true|mostly true|fact check|fact-check|debunked)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return normalize_text(cleaned)


def dedupe_examples(examples: list[ClaimDatasetExample]) -> list[ClaimDatasetExample]:
    seen: set[tuple[int, str]] = set()
    deduped: list[ClaimDatasetExample] = []
    for example in examples:
        key = (example.label, example.text.lower())
        if key in seen or len(example.text) < 18:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def balance_examples(
    examples: list[ClaimDatasetExample],
    *,
    max_per_source: int,
    seed: int,
) -> list[ClaimDatasetExample]:
    rng = random.Random(seed)
    by_source: dict[str, list[ClaimDatasetExample]] = {}
    for example in examples:
        by_source.setdefault(example.source, []).append(example)
    limited: list[ClaimDatasetExample] = []
    for source_examples in by_source.values():
        rows = list(source_examples)
        rng.shuffle(rows)
        limited.extend(rows[:max_per_source])
    positives = [example for example in limited if example.label == 1]
    negatives = [example for example in limited if example.label == 0]
    target = min(len(positives), len(negatives))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:target] + negatives[:target]


def collect_news_negatives(client: httpx.Client, *, items_per_feed: int) -> list[ClaimDatasetExample]:
    examples: list[ClaimDatasetExample] = []
    for source_name, feed_url in NEWS_RSS_FEEDS:
        response = client.get(feed_url)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        for index, item in enumerate(root.findall("./channel/item")[:items_per_feed]):
            title = item.findtext("title", default="")
            description = item.findtext("description", default="")
            link = item.findtext("link", default=feed_url)
            text = normalize_text(title, BeautifulSoup(description, "html.parser").get_text(" ", strip=True))
            if not text:
                continue
            examples.append(
                ClaimDatasetExample(
                    text=text,
                    label=0,
                    source=source_name,
                    source_url=link,
                    item_id=f"{source_name}:{index}",
                )
            )
    return examples


def collect_snopes_examples(client: httpx.Client, *, limit_per_rating: int) -> list[ClaimDatasetExample]:
    examples: list[ClaimDatasetExample] = []
    for source_name, rating_url in SNOPES_FALSE_URLS:
        is_false = 1 if "positive" in source_name else 0
        page = client.get(rating_url)
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        links = []
        for anchor in soup.select("a[href*='/fact-check/']"):
            href = str(anchor.get("href", "")).strip()
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin("https://www.snopes.com", href)
            if "/rating/" in href:
                continue
            links.append(href)
        for link in list(dict.fromkeys(links))[:limit_per_rating]:
            article = client.get(link)
            if article.status_code != 200:
                continue
            article_soup = BeautifulSoup(article.text, "html.parser")
            title = article_soup.select_one("h1")
            description = article_soup.select_one("meta[name=description]")
            text = strip_verdict_markers(
                normalize_text(
                    title.get_text(" ", strip=True) if title else "",
                    description.get("content", "") if description else "",
                )
            )
            if not text:
                continue
            examples.append(
                ClaimDatasetExample(
                    text=text,
                    label=is_false,
                    source=source_name,
                    source_url=link,
                    item_id=link,
                )
            )
    return examples


def collect_politifact_examples(client: httpx.Client, *, pages_per_ruling: int) -> list[ClaimDatasetExample]:
    examples: list[ClaimDatasetExample] = []
    for source_name, ruling, label in POLITIFACT_RULINGS:
        for page_number in range(1, pages_per_ruling + 1):
            list_url = f"https://www.politifact.com/factchecks/list/?page={page_number}&ruling={ruling}"
            listing = client.get(list_url)
            listing.raise_for_status()
            relative_links = sorted(set(re.findall(r"/factchecks/\d{4}/[a-z]{3}/\d{2}/[^\"']+/", listing.text)))
            article_links = [urljoin("https://www.politifact.com", link) for link in relative_links]
            for link in article_links:
                article = client.get(link)
                if article.status_code != 200:
                    continue
                soup = BeautifulSoup(article.text, "html.parser")
                title = soup.select_one("h1")
                description = soup.select_one("meta[name=description]")
                text = strip_verdict_markers(
                    normalize_text(
                        title.get_text(" ", strip=True) if title else "",
                        description.get("content", "") if description else "",
                    )
                )
                if not text:
                    continue
                examples.append(
                    ClaimDatasetExample(
                        text=text,
                        label=label,
                        source=source_name,
                        source_url=link,
                        item_id=link,
                    )
                )
    return examples


def synthetic_claim_examples() -> list[ClaimDatasetExample]:
    positives = (
        "Immigrants are secretly receiving unlimited debit cards from the government",
        "A miracle herb cures cancer in 24 hours and doctors are hiding it",
        "Voting machines switched millions of votes overnight without evidence",
        "A celebrity endorsed a crypto giveaway that doubles every payment instantly",
        "A city banned all private car ownership effective next week",
    )
    negatives = (
        "The central bank held interest rates steady after its April meeting",
        "Researchers released a benchmark for graph retrieval systems",
        "The ministry published updated flood response guidance",
        "The court scheduled oral arguments for the antitrust appeal",
        "Engineers documented rollout checks for the new search service",
    )
    examples: list[ClaimDatasetExample] = []
    for index, text in enumerate(positives):
        examples.append(
            ClaimDatasetExample(
                text=text,
                label=1,
                source="synthetic_false_positive",
                source_url="synthetic://false",
                item_id=f"false:{index}",
            )
        )
    for index, text in enumerate(negatives):
        examples.append(
            ClaimDatasetExample(
                text=text,
                label=0,
                source="synthetic_true_negative",
                source_url="synthetic://true",
                item_id=f"true:{index}",
            )
        )
    return examples


def collect_dataset(
    output_path: Path,
    *,
    news_items_per_feed: int,
    factcheck_pages_per_ruling: int,
    max_per_source: int,
    seed: int,
) -> dict[str, int]:
    client = httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "jakal-search-falsehood-training/0.1"},
    )
    try:
        examples = []
        examples.extend(collect_news_negatives(client, items_per_feed=news_items_per_feed))
        examples.extend(collect_snopes_examples(client, limit_per_rating=max(20, factcheck_pages_per_ruling * 20)))
        examples.extend(collect_politifact_examples(client, pages_per_ruling=factcheck_pages_per_ruling))
        examples.extend(synthetic_claim_examples())
    finally:
        client.close()

    examples = dedupe_examples(examples)
    examples = balance_examples(examples, max_per_source=max_per_source, seed=seed)
    random.Random(seed).shuffle(examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(example.to_json() for example in examples), encoding="utf-8")
    return {
        "total": len(examples),
        "false_positive": sum(example.label == 1 for example in examples),
        "true_negative": sum(example.label == 0 for example in examples),
        "sources": len({example.source for example in examples}),
        "max_per_source": max_per_source,
    }


def load_examples(dataset_path: Path) -> list[ClaimDatasetExample]:
    return [
        ClaimDatasetExample(**json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stratified_split(
    examples: list[ClaimDatasetExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[ClaimDatasetExample], list[ClaimDatasetExample]]:
    rng = random.Random(seed)
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_val = max(1, int(len(positives) * val_ratio))
    neg_val = max(1, int(len(negatives) * val_ratio))
    train = positives[pos_val:] + negatives[neg_val:]
    val = positives[:pos_val] + negatives[:neg_val]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def evaluate_model(model: ClaimFalsehoodMLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, TrainingMetrics]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32)))
        probabilities = torch.sigmoid(logits).numpy()
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
    return best_threshold, TrainingMetrics(
        train_size=0,
        val_size=len(labels),
        threshold=best_threshold,
        val_accuracy=float(accuracy),
        val_precision=float(precision),
        val_recall=float(recall),
        val_f1=float(f1),
    )


def train_falsehood_head(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    embedding_device: str,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    val_ratio: float,
    seed: int,
) -> tuple[Path, TrainingMetrics]:
    examples = load_examples(dataset_path)
    train_examples, val_examples = stratified_split(examples, val_ratio=val_ratio, seed=seed)
    embedder = SentenceTransformerEmbedder(model_name, device=embedding_device)
    train_x = embedder.embed([example.text for example in train_examples]).astype(np.float32)
    val_x = embedder.embed([example.text for example in val_examples]).astype(np.float32)
    train_y = np.asarray([example.label for example in train_examples], dtype=np.float32)
    val_y = np.asarray([example.label for example in val_examples], dtype=np.float32)

    torch.manual_seed(seed)
    model = ClaimFalsehoodMLP(train_x.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = max(float(np.sum(train_y == 1.0)), 1.0)
    negatives = max(float(np.sum(train_y == 0.0)), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))

    train_inputs = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_y)
    permutation = np.arange(len(train_examples))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        rng.shuffle(permutation)
        for start in range(0, len(permutation), batch_size):
            indexes = permutation[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(train_inputs[indexes])
            loss = loss_fn(logits, train_targets[indexes])
            loss.backward()
            optimizer.step()

    threshold, eval_metrics = evaluate_model(model, val_x, val_y)
    metrics = TrainingMetrics(
        train_size=len(train_examples),
        val_size=len(val_examples),
        threshold=threshold,
        val_accuracy=eval_metrics.val_accuracy,
        val_precision=eval_metrics.val_precision,
        val_recall=eval_metrics.val_recall,
        val_f1=eval_metrics.val_f1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace("/", "__")
    model_path = output_dir / f"claim_falsehood_head_{safe_model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": hidden_dim,
            "threshold": threshold,
            "model_name": model_name,
        },
        model_path,
    )
    model_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "embedding_device": embedding_device,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect claim falsehood data and train a false-claim classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect false-claim training data.")
    collect_parser.add_argument("--dataset-out", type=Path, default=Path("outputs") / "datasets" / "claim_falsehood_dataset.jsonl")
    collect_parser.add_argument("--news-items-per-feed", type=int, default=120)
    collect_parser.add_argument("--factcheck-pages-per-ruling", type=int, default=4)
    collect_parser.add_argument("--max-per-source", type=int, default=220)
    collect_parser.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train", help="Train false-claim head from collected dataset.")
    train_parser.add_argument("--dataset-path", type=Path, default=Path("outputs") / "datasets" / "claim_falsehood_dataset.jsonl")
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "models")
    train_parser.add_argument("--model-name", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    train_parser.add_argument("--embedding-device", default="directml")
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=0)

    run_parser = subparsers.add_parser("run", help="Collect and train in one command.")
    run_parser.add_argument("--dataset-out", type=Path, default=Path("outputs") / "datasets" / "claim_falsehood_dataset.jsonl")
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "models")
    run_parser.add_argument("--news-items-per-feed", type=int, default=120)
    run_parser.add_argument("--factcheck-pages-per-ruling", type=int, default=4)
    run_parser.add_argument("--max-per-source", type=int, default=220)
    run_parser.add_argument("--model-name", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    run_parser.add_argument("--embedding-device", default="directml")
    run_parser.add_argument("--hidden-dim", type=int, default=128)
    run_parser.add_argument("--epochs", type=int, default=8)
    run_parser.add_argument("--learning-rate", type=float, default=5e-4)
    run_parser.add_argument("--weight-decay", type=float, default=1e-4)
    run_parser.add_argument("--batch-size", type=int, default=64)
    run_parser.add_argument("--val-ratio", type=float, default=0.2)
    run_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "collect":
        counts = collect_dataset(
            args.dataset_out,
            news_items_per_feed=args.news_items_per_feed,
            factcheck_pages_per_ruling=args.factcheck_pages_per_ruling,
            max_per_source=args.max_per_source,
            seed=args.seed,
        )
        print(json.dumps({"dataset_path": str(args.dataset_out), "counts": counts}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train":
        model_path, metrics = train_falsehood_head(
            args.dataset_path,
            args.output_dir,
            model_name=args.model_name,
            embedding_device=args.embedding_device,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(json.dumps({"model_path": str(model_path), "metrics": asdict(metrics)}, ensure_ascii=False, indent=2))
        return 0

    counts = collect_dataset(
        args.dataset_out,
        news_items_per_feed=args.news_items_per_feed,
        factcheck_pages_per_ruling=args.factcheck_pages_per_ruling,
        max_per_source=args.max_per_source,
        seed=args.seed,
    )
    model_path, metrics = train_falsehood_head(
        args.dataset_out,
        args.output_dir,
        model_name=args.model_name,
        embedding_device=args.embedding_device,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "dataset_path": str(args.dataset_out),
                "counts": counts,
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
