from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import torch
from bs4 import BeautifulSoup
from torch import nn

from .config import TrustConfig
from .embedding import SentenceTransformerEmbedder
from .trust import EmbeddingTrustMLP, build_trust_feature_vector, resolve_source_profile


ARXIV_API_URL = "https://export.arxiv.org/api/query"
DOC_SOURCE_INDEXES = (
    ("python_docs_positive", "https://docs.python.org/3/contents.html"),
    ("sklearn_docs_positive", "https://scikit-learn.org/stable/user_guide.html"),
    ("pandas_docs_positive", "https://pandas.pydata.org/docs/"),
    ("requests_docs_positive", "https://requests.readthedocs.io/en/latest/"),
)
NEWS_RSS_FEEDS = (
    ("bbc_world_news_positive", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("bbc_technology_news_positive", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("npr_news_positive", "https://www.npr.org/rss/rss.php?id=1001"),
)
SMS_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
YOUTUBE_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
SPAMASSASSIN_INDEX_URL = "https://spamassassin.apache.org/old/publiccorpus/"


@dataclass(slots=True)
class TrustDatasetExample:
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


def collect_dataset(
    output_path: Path,
    *,
    arxiv_per_category: int = 300,
    python_docs_pages: int = 200,
    spamassassin_limit: int = 1200,
    news_items_per_feed: int = 80,
    max_per_source: int = 280,
    balance_labels: bool = True,
    random_seed: int = 0,
) -> dict[str, int]:
    client = httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "jakal-search-trust-training/0.1"},
    )
    try:
        positives = []
        if arxiv_per_category > 0:
            positives.extend(collect_arxiv_examples(client, per_category=arxiv_per_category))
        positives.extend(collect_documentation_examples(client, max_pages=python_docs_pages))
        positives.extend(collect_news_positive_examples(client, items_per_feed=news_items_per_feed))

        ham_positives, sms_negatives = collect_sms_examples(client)
        youtube_positives, youtube_negatives = collect_youtube_examples(client)
        spamassassin_positives, spamassassin_negatives = collect_spamassassin_examples(client, limit=spamassassin_limit)

        positives.extend(ham_positives)
        positives.extend(youtube_positives)
        positives.extend(spamassassin_positives)

        negatives = []
        negatives.extend(sms_negatives)
        negatives.extend(youtube_negatives)
        negatives.extend(spamassassin_negatives)

        hard_positives, hard_negatives = generate_hard_synthetic_examples()
        positives.extend(hard_positives)
        negatives.extend(hard_negatives)
    finally:
        client.close()

    examples = dedupe_examples([*positives, *negatives])
    examples = balance_examples(
        examples,
        max_per_source=max_per_source,
        balance_labels=balance_labels,
        seed=random_seed,
    )
    rng = random.Random(random_seed)
    rng.shuffle(examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(example.to_json() for example in examples), encoding="utf-8")
    label_counts = {
        "total": len(examples),
        "positive": sum(example.label == 1 for example in examples),
        "negative": sum(example.label == 0 for example in examples),
    }
    return {
        **label_counts,
        "sources": len({example.source for example in examples}),
        "max_per_source": max_per_source,
    }


def collect_news_positive_examples(client: httpx.Client, *, items_per_feed: int) -> list[TrustDatasetExample]:
    examples: list[TrustDatasetExample] = []
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
                TrustDatasetExample(
                    text=text,
                    label=1,
                    source=source_name,
                    source_url=link,
                    item_id=f"{source_name}:{index}",
                )
            )
    return examples


def collect_arxiv_examples(client: httpx.Client, *, per_category: int) -> list[TrustDatasetExample]:
    categories = ("cs.AI", "cs.CL", "cs.IR", "cs.SE", "cs.LG")
    examples: list[TrustDatasetExample] = []
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    for category in categories:
        for start in range(0, per_category, 100):
            batch_size = min(100, per_category - start)
            response = get_with_backoff(
                client,
                ARXIV_API_URL,
                params={"search_query": f"cat:{category}", "start": start, "max_results": batch_size},
                attempts=5,
                base_delay=3.0,
            )
            root = ET.fromstring(response.text)
            for entry in root.findall("atom:entry", namespace):
                title = entry.findtext("atom:title", default="", namespaces=namespace)
                summary = entry.findtext("atom:summary", default="", namespaces=namespace)
                item_id = entry.findtext("atom:id", default="", namespaces=namespace)
                text = normalize_text(title, summary)
                if not text:
                    continue
                examples.append(
                    TrustDatasetExample(
                        text=text,
                        label=1,
                        source="arxiv_positive",
                        source_url=item_id,
                        item_id=item_id or f"arxiv:{category}:{start}:{len(examples)}",
                    )
                )
            time.sleep(1.0)
    return examples


def collect_documentation_examples(client: httpx.Client, *, max_pages: int) -> list[TrustDatasetExample]:
    examples: list[TrustDatasetExample] = []
    per_source = max(1, max_pages // len(DOC_SOURCE_INDEXES))
    remainder = max_pages % len(DOC_SOURCE_INDEXES)

    for index, (source_name, root_url) in enumerate(DOC_SOURCE_INDEXES):
        source_limit = per_source + (1 if index < remainder else 0)
        examples.extend(collect_docs_from_index(client, source_name, root_url, source_limit))
    return examples


def collect_docs_from_index(
    client: httpx.Client,
    source_name: str,
    root_url: str,
    max_pages: int,
) -> list[TrustDatasetExample]:
    response = client.get(root_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    queue: list[str] = []
    seen_urls = {root_url}

    for anchor in soup.select("a"):
        href = anchor.get("href")
        if not href:
            continue
        url = normalize_docs_url(urljoin(root_url, href), root_url=root_url)
        if url is None or url in seen_urls:
            continue
        seen_urls.add(url)
        queue.append(url)

    examples: list[TrustDatasetExample] = []
    while queue and len(examples) < max_pages:
        url = queue.pop(0)
        page = client.get(url)
        if page.status_code != 200:
            continue
        page_soup = BeautifulSoup(page.text, "html.parser")
        title = page_soup.title.get_text(" ", strip=True) if page_soup.title else ""
        paragraphs = page_soup.select("div.body p, main p, article p")
        snippet = ""
        for paragraph in paragraphs:
            candidate = paragraph.get_text(" ", strip=True)
            if len(candidate) >= 60:
                snippet = candidate
                break
        text = normalize_text(title, snippet)
        if text:
            examples.append(
                TrustDatasetExample(
                    text=text,
                    label=1,
                    source=source_name,
                    source_url=url,
                    item_id=url,
                )
            )

        for anchor in page_soup.select("a"):
            href = anchor.get("href")
            if not href:
                continue
            next_url = normalize_docs_url(urljoin(url, href), root_url=root_url)
            if next_url is None or next_url in seen_urls:
                continue
            seen_urls.add(next_url)
            queue.append(next_url)
    return examples


def normalize_docs_url(url: str, *, root_url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    root = urlparse(root_url)
    if parsed.netloc != root.netloc:
        return None
    if not parsed.path.startswith(root.path.rstrip("/") or "/"):
        return None
    if parsed.path.endswith((".txt", ".inv", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".zip")):
        return None
    if "/_static/" in parsed.path or "/_images/" in parsed.path or "/genindex" in parsed.path:
        return None
    clean = parsed._replace(fragment="", query="")
    return clean.geturl()


def collect_sms_examples(client: httpx.Client) -> tuple[list[TrustDatasetExample], list[TrustDatasetExample]]:
    response = client.get(SMS_SPAM_URL)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        raw = archive.read("SMSSpamCollection").decode("utf-8", errors="replace")
    positives: list[TrustDatasetExample] = []
    negatives: list[TrustDatasetExample] = []
    for index, line in enumerate(raw.splitlines()):
        if "\t" not in line:
            continue
        label, message = line.split("\t", 1)
        text = normalize_text(message)
        if not text:
            continue
        target = negatives if label.strip().lower() == "spam" else positives
        source_name = "sms_spam_negative" if label.strip().lower() == "spam" else "sms_ham_positive"
        target.append(
            TrustDatasetExample(
                text=text,
                label=0 if label.strip().lower() == "spam" else 1,
                source=source_name,
                source_url=SMS_SPAM_URL,
                item_id=f"sms:{label.strip().lower()}:{index}",
            )
        )
    return positives, negatives


def collect_youtube_examples(client: httpx.Client) -> tuple[list[TrustDatasetExample], list[TrustDatasetExample]]:
    response = client.get(YOUTUBE_SPAM_URL)
    response.raise_for_status()
    positives: list[TrustDatasetExample] = []
    negatives: list[TrustDatasetExample] = []
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with archive.open(name) as handle:
                rows = csv.DictReader((line.decode("utf-8", errors="replace") for line in handle))
                for index, row in enumerate(rows):
                    text = normalize_text(row.get("CONTENT", ""))
                    if not text:
                        continue
                    is_spam = str(row.get("CLASS", "")).strip() == "1"
                    target = negatives if is_spam else positives
                    source_name = "youtube_spam_negative" if is_spam else "youtube_ham_positive"
                    target.append(
                        TrustDatasetExample(
                            text=text,
                            label=0 if is_spam else 1,
                            source=source_name,
                            source_url=YOUTUBE_SPAM_URL,
                            item_id=f"{name}:{index}",
                        )
                    )
    return positives, negatives


def collect_spamassassin_examples(
    client: httpx.Client,
    *,
    limit: int,
) -> tuple[list[TrustDatasetExample], list[TrustDatasetExample]]:
    index_page = client.get(SPAMASSASSIN_INDEX_URL)
    index_page.raise_for_status()
    soup = BeautifulSoup(index_page.text, "html.parser")
    archive_urls: list[tuple[str, int]] = []
    for anchor in soup.select("a"):
        href = str(anchor.get("href", "")).strip()
        if not href.endswith((".tar.bz2", ".tar.gz")):
            continue
        lowered = href.lower()
        if "spam" not in lowered and "easy_ham" not in lowered:
            continue
        label = 0 if "spam" in lowered and "easy_ham" not in lowered else 1
        archive_urls.append((urljoin(SPAMASSASSIN_INDEX_URL, href), label))

    positives: list[TrustDatasetExample] = []
    negatives: list[TrustDatasetExample] = []
    for archive_url, label in archive_urls[:4]:
        response = client.get(archive_url)
        response.raise_for_status()
        mode = "r:gz" if archive_url.endswith(".gz") else "r:bz2"
        with tarfile.open(fileobj=BytesIO(response.content), mode=mode) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                file_handle = archive.extractfile(member)
                if file_handle is None:
                    continue
                payload = file_handle.read()
                text = extract_email_text(payload)
                if not text:
                    continue
                target = negatives if label == 0 else positives
                source_name = "spamassassin_negative" if label == 0 else "spamassassin_ham_positive"
                target.append(
                    TrustDatasetExample(
                        text=text,
                        label=label,
                        source=source_name,
                        source_url=archive_url,
                        item_id=f"{archive_url}:{member.name}",
                    )
                )
                if len(positives) >= limit and len(negatives) >= limit:
                    return positives[:limit], negatives[:limit]
    return positives[:limit], negatives[:limit]


def get_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, object] | None = None,
    attempts: int,
    base_delay: float,
) -> httpx.Response:
    last_response: httpx.Response | None = None
    for attempt in range(attempts):
        response = client.get(url, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        last_response = response
        time.sleep(base_delay * (attempt + 1))
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url}")


def extract_email_text(payload: bytes) -> str:
    message = BytesParser(policy=policy.default).parsebytes(payload)
    subject = str(message.get("subject", "")).strip()
    body = ""
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() != "text":
                continue
            try:
                candidate = part.get_content()
            except Exception:
                continue
            if candidate:
                body = str(candidate)
                break
    else:
        try:
            body = str(message.get_content())
        except Exception:
            body = payload.decode("utf-8", errors="replace")
    body = " ".join(body.split())[:600]
    return normalize_text(subject, body)


def dedupe_examples(examples: list[TrustDatasetExample]) -> list[TrustDatasetExample]:
    seen: set[tuple[int, str]] = set()
    deduped: list[TrustDatasetExample] = []
    for example in examples:
        key = (example.label, example.text.lower())
        if key in seen or len(example.text) < 24:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def balance_examples(
    examples: list[TrustDatasetExample],
    *,
    max_per_source: int,
    balance_labels: bool,
    seed: int,
) -> list[TrustDatasetExample]:
    rng = random.Random(seed)
    by_source: dict[str, list[TrustDatasetExample]] = {}
    for example in examples:
        by_source.setdefault(example.source, []).append(example)

    limited: list[TrustDatasetExample] = []
    for source_examples in by_source.values():
        rows = list(source_examples)
        rng.shuffle(rows)
        limited.extend(rows[:max_per_source])

    if not balance_labels:
        return limited

    positives = [example for example in limited if example.label == 1]
    negatives = [example for example in limited if example.label == 0]
    target = min(len(positives), len(negatives))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:target] + negatives[:target]


def generate_hard_synthetic_examples() -> tuple[list[TrustDatasetExample], list[TrustDatasetExample]]:
    hard_positives: list[TrustDatasetExample] = []
    hard_negatives: list[TrustDatasetExample] = []

    positive_titles = (
        "conspiracy detection benchmark report",
        "spam campaign incident analysis",
        "urgent phishing bulletin for maintainers",
        "miracle-claim moderation taxonomy",
        "giveaway scam dataset card",
    )
    positive_bodies = (
        "peer reviewed study with evaluation protocol, ablations, and cited baselines",
        "official incident response guide with remediation steps, examples, and rollback notes",
        "maintainer playbook documenting evidence, failure modes, and mitigation tradeoffs",
        "academic analysis with labeled examples, annotation guidance, and reproducible metrics",
    )
    negative_titles = (
        "official documentation for guaranteed ranking wins",
        "peer reviewed miracle traffic formula",
        "academic secret loophole handbook",
        "government-style instant authority guide",
        "maintainer approved giveaway accelerator",
    )
    negative_bodies = (
        "urgent action required for instant results with hidden methods and no evidence",
        "secret system that bypasses rules, guarantees growth, and reveals the truth they hide",
        "exclusive loophole for free traffic and miracle outcomes with copied testimonials",
        "limited offer combining official sounding language with unverifiable claims and hacks",
    )

    item_id = 0
    for title in positive_titles:
        for body in positive_bodies:
            text = normalize_text(title, body)
            hard_positives.append(
                TrustDatasetExample(
                    text=text,
                    label=1,
                    source="synthetic_hard_positive",
                    source_url="synthetic://hard-positive",
                    item_id=f"synthetic-positive:{item_id}",
                )
            )
            item_id += 1

    item_id = 0
    for title in negative_titles:
        for body in negative_bodies:
            text = normalize_text(title, body)
            hard_negatives.append(
                TrustDatasetExample(
                    text=text,
                    label=0,
                    source="synthetic_hard_negative",
                    source_url="synthetic://hard-negative",
                    item_id=f"synthetic-negative:{item_id}",
                )
            )
            item_id += 1

    return hard_positives, hard_negatives


def load_examples(dataset_path: Path) -> list[TrustDatasetExample]:
    examples: list[TrustDatasetExample] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        examples.append(TrustDatasetExample(**payload))
    return examples


def stratified_split(
    examples: list[TrustDatasetExample],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[TrustDatasetExample], list[TrustDatasetExample]]:
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


def embed_texts(texts: list[str], *, embedder: SentenceTransformerEmbedder) -> np.ndarray:
    return embedder.embed(texts)


def build_training_matrix(
    examples: list[TrustDatasetExample],
    embeddings: np.ndarray,
    *,
    config: TrustConfig,
) -> np.ndarray:
    rows = []
    for example, embedding in zip(examples, embeddings):
        profile = resolve_source_profile(config, example.source_url)
        rows.append(build_trust_feature_vector(embedding, profile))
    return np.asarray(rows, dtype=np.float32)


def train_trust_head(
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

    trust_config = TrustConfig()
    train_embeddings = embed_texts([example.text for example in train_examples], embedder=embedder)
    val_embeddings = embed_texts([example.text for example in val_examples], embedder=embedder)
    train_x = build_training_matrix(train_examples, train_embeddings, config=trust_config)
    val_x = build_training_matrix(val_examples, val_embeddings, config=trust_config)
    train_y = np.asarray([example.label for example in train_examples], dtype=np.float32)
    val_y = np.asarray([example.label for example in val_examples], dtype=np.float32)

    torch.manual_seed(seed)
    model = EmbeddingTrustMLP(train_x.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = max(float(np.sum(train_y == 1.0)), 1.0)
    negatives = max(float(np.sum(train_y == 0.0)), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_inputs = torch.from_numpy(train_x.astype(np.float32))
    train_targets = torch.from_numpy(train_y.astype(np.float32))
    permutation = np.arange(len(train_examples))
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        rng.shuffle(permutation)
        for start in range(0, len(permutation), batch_size):
            indexes = permutation[start : start + batch_size]
            batch_inputs = train_inputs[indexes]
            batch_targets = train_targets[indexes]
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_targets)
            loss.backward()
            optimizer.step()

    threshold, metrics = evaluate_model(model, val_x, val_y)
    metrics = TrainingMetrics(
        train_size=len(train_examples),
        val_size=metrics.val_size,
        threshold=metrics.threshold,
        val_accuracy=metrics.val_accuracy,
        val_precision=metrics.val_precision,
        val_recall=metrics.val_recall,
        val_f1=metrics.val_f1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace("/", "__")
    model_path = output_dir / f"trust_head_{safe_model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": int(hidden_dim),
            "threshold": float(threshold),
            "model_name": model_name,
        },
        model_path,
    )
    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text(
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


def evaluate_model(model: EmbeddingTrustMLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, TrainingMetrics]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect trust data and train a supervised trust MLP on sentence-transformer embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect trust training data from official sources.")
    collect_parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("outputs") / "datasets" / "trust_dataset.jsonl",
    )
    collect_parser.add_argument("--arxiv-per-category", type=int, default=300)
    collect_parser.add_argument("--python-docs-pages", type=int, default=200)
    collect_parser.add_argument("--spamassassin-limit", type=int, default=1200)
    collect_parser.add_argument("--news-items-per-feed", type=int, default=80)
    collect_parser.add_argument("--max-per-source", type=int, default=280)
    collect_parser.add_argument("--no-balance-labels", action="store_true")
    collect_parser.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train", help="Train a supervised trust head from a collected dataset.")
    train_parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("outputs") / "datasets" / "trust_dataset.jsonl",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "models",
    )
    train_parser.add_argument("--model-name", default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    train_parser.add_argument("--embedding-device", default="directml")
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=0)

    run_parser = subparsers.add_parser("run", help="Collect data and train in one command.")
    run_parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("outputs") / "datasets" / "trust_dataset.jsonl",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "models",
    )
    run_parser.add_argument("--arxiv-per-category", type=int, default=300)
    run_parser.add_argument("--python-docs-pages", type=int, default=200)
    run_parser.add_argument("--spamassassin-limit", type=int, default=1200)
    run_parser.add_argument("--news-items-per-feed", type=int, default=80)
    run_parser.add_argument("--max-per-source", type=int, default=280)
    run_parser.add_argument("--no-balance-labels", action="store_true")
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
            arxiv_per_category=args.arxiv_per_category,
            python_docs_pages=args.python_docs_pages,
            spamassassin_limit=args.spamassassin_limit,
            news_items_per_feed=args.news_items_per_feed,
            max_per_source=args.max_per_source,
            balance_labels=not args.no_balance_labels,
            random_seed=args.seed,
        )
        print(json.dumps({"dataset_path": str(args.dataset_out), "counts": counts}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train":
        model_path, metrics = train_trust_head(
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
        arxiv_per_category=args.arxiv_per_category,
        python_docs_pages=args.python_docs_pages,
        spamassassin_limit=args.spamassassin_limit,
        news_items_per_feed=args.news_items_per_feed,
        max_per_source=args.max_per_source,
        balance_labels=not args.no_balance_labels,
        random_seed=args.seed,
    )
    model_path, metrics = train_trust_head(
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
