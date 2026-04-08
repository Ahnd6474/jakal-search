# jakal-search

Recursive web search for evidence collection, topic expansion, and machine-readable news records.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What is jakal-search?](#what-is-jakal-search)
- [Why this exists](#why-this-exists)
- [Search pipeline](#search-pipeline)
- [Why the theme graph matters](#why-the-theme-graph-matters)
- [Embeddings](#embeddings)
- [Providers](#providers)
- [Historical cutoffs](#historical-cutoffs)
- [Output formats](#output-formats)
- [CLI reference](#cli-reference)
- [Examples](#examples)
- [Training and tuning utilities](#training-and-tuning-utilities)
- [Desktop GUI](#desktop-gui)
- [Deployment](#deployment)
- [License](#license)

## Installation

Install the published package:

```bash
pip install jakal-search
jakal-search "graph search systems"
```

If you want transformer embeddings instead of the hashing fallback:

```bash
pip install "jakal-search[models]"
```

On Windows, DirectML is the practical GPU path for Intel integrated graphics:

```bash
pip install "jakal-search[models,directml]"
jakal-search "graph search systems" --device directml
```

For local development:

```bash
pip install -e .[dev]
```

For local development with transformer support:

```bash
pip install -e .[dev,models]
```

## Quick start

Run a normal exploratory search:

```bash
python -m jakal_search "graph search systems" --format report
```

Collect market-news records at a strict historical cutoff:

```bash
python -m jakal_search "AAPL earnings" \
  --stock-news \
  --as-of 2026-04-08T09:00:00Z \
  --format records
```

Write the full search tree to disk:

```bash
python -m jakal_search "semiconductor AI demand" --json-out outputs/tree.json
```

## What is jakal-search?

`jakal-search` is a Python package and CLI for recursive web search. It starts from one query, collects documents, scores them, groups them into subtopics, generates new topic queries, and searches again.

The current pipeline is aimed at evidence gathering rather than chat-style answering. It can still print human-readable reports, but the more important mode is `records`, which emits structured per-document output for downstream models.

## Why this exists

This project is trying to do something narrower than "general AI search." It is meant to build a reproducible search stage for pipelines like market-state estimation, where you care about:

- strict `as_of` cutoffs
- source and trust signals
- passage-level evidence
- recursive topic expansion
- machine-readable output instead of prose

That is why the engine keeps the search layer deterministic and pushes richer modeling into later stages.

## Search pipeline

At a high level, each node in the search tree does this:

1. Search the web with the current query.
2. Fetch the top landing pages and extract readable text.
3. Split pages into passages and rerank documents with lexical, dense, provider, freshness, and entity signals.
4. Filter low-trust or high-risk material.
5. Build theme tokens from the surviving document embeddings.
6. Run a directed fully connected theme graph with no self-edges.
7. Use the propagated theme states to attend back over documents.
8. Decode theme-specific topic labels and queries.
9. Search again with those decoded topic queries.

The current theme layer is not just "keyword extraction." It keeps a latent graph over abstract theme tokens and uses the graph output to produce the next round of topic queries.

## Why the theme graph matters

The graph is there to let indirect relationships survive the search stage.

Examples:

- AI demand -> semiconductors
- war ending -> reconstruction -> construction
- higher oil -> airlines under pressure

The engine does not hard-code those examples as labels. Instead, it builds theme vectors from documents, propagates them through a directed graph, and decodes new topic queries from the propagated states.

## Embeddings

Transformer embeddings are optional. If `sentence-transformers` is not installed, the engine falls back to a lightweight hashing embedder.

The default transformer model is:

```text
sentence-transformers/paraphrase-MiniLM-L3-v2
```

That choice is deliberate. The retrieval side needs to stay light enough to run repeatedly during recursive search.

## Providers

The CLI supports multiple free provider packs:

| Pack | Intended use |
|------|--------------|
| `general` | General web search |
| `technical` | Docs, engineering, software topics |
| `research` | Paper-like and research-heavy queries |
| `news` | General current-events queries |
| `market_news` | Stock and company news collection |

The `market_news` pack currently combines free sources such as Google News RSS, GDELT, and DuckDuckGo HTML.

Use it directly:

```bash
python -m jakal_search "NVDA guidance" --provider-pack market_news
```

Or use the shortcut:

```bash
python -m jakal_search "NVDA guidance" --stock-news
```

## Historical cutoffs

If you are building training data or running backtests, use `--as-of`.

```bash
python -m jakal_search "TSMC capex" --as-of 2026-04-08T09:00:00Z --format records
```

This does two things:

- keeps only documents published at or before the cutoff
- computes freshness and recency relative to that cutoff, not relative to "now"

Date-only articles are treated conservatively when the cutoff is intraday.

## Output formats

The CLI supports these output modes:

| Format | What it is for |
|--------|----------------|
| `report` | Human-readable summary |
| `urls` | URL list only |
| `tree` | Search tree overview |
| `json` | Full internal search tree |
| `answer` | Evidence-first answer summary |
| `records` | Machine-readable per-document records |

The `records` mode is the one to use if another model is going to read the output.

Each record includes fields such as:

- query and `as_of`
- title, snippet, content, evidence
- `published_at` and precision
- retrieval, trust, freshness, and entity scores
- theme memberships
- topic memberships
- `embedding_text` for downstream encoders

## CLI reference

Common options:

| Option | Description |
|--------|-------------|
| `--max-depth` | Maximum recursive depth |
| `--max-nodes` | Global node budget |
| `--frontier-width` | How many pending nodes stay in the frontier |
| `--results-per-query` | Provider results per search |
| `--fetch-top-k` | How many results get landing-page enrichment |
| `--no-page-fetch` | Skip landing-page fetching |
| `--provider-pack` | Choose `general`, `technical`, `research`, `news`, or `market_news` |
| `--stock-news` | Shortcut for market-news settings |
| `--as-of` | Strict historical cutoff |
| `--format` | Output mode |
| `--json-out` | Write full JSON tree to a file |
| `--device` | `auto`, `cpu`, `cuda`, `mps`, `xpu`, or `directml` |
| `--trust-model` | Trained trust head |
| `--branch-model` | Trained branch decision head |
| `--topic-reranker-model` | Trained topic/query reranker |
| `--claim-model` | Trained false-claim head |

## Examples

### General search with topic expansion

```bash
python -m jakal_search "graph search systems" --max-depth 3 --format tree
```

### Market-news collection for downstream modeling

```bash
python -m jakal_search "semiconductor AI demand" \
  --stock-news \
  --as-of 2026-04-08T09:00:00Z \
  --format records \
  --json-out outputs/semiconductor-tree.json
```

### Disable page fetch to run faster

```bash
python -m jakal_search "bank regulation" --no-page-fetch --format report
```

### Increase landing-page enrichment

```bash
python -m jakal_search "AAPL earnings" --stock-news --fetch-top-k 10 --format records
```

## Training and tuning utilities

The package ships with several local utilities:

- `jakal-search-tune`
- `jakal-search-trust`
- `jakal-search-branch`
- `jakal-search-reranker`
- `jakal-search-falsehood`

Examples:

```bash
python -m jakal_search.tuning
python -m jakal_search.trust_training run --embedding-device directml
```

If you use trained heads at runtime:

```bash
python -m jakal_search "graph search systems" \
  --device directml \
  --trust-model outputs/models/trust_head_sentence-transformers__paraphrase-MiniLM-L3-v2.pt \
  --claim-model outputs/models/claim_falsehood_head_sentence-transformers__paraphrase-MiniLM-L3-v2.pt \
  --branch-model outputs/models/branch_head.pt \
  --topic-reranker-model outputs/models/topic_reranker_head.pt
```

## Desktop GUI

The desktop app lives in `desktop/` and uses Tauri plus React. It is set up as a blind pairwise comparison UI over search results.

Start it like this:

```bash
cd desktop
npm install
npm run tauri dev
```

The GUI calls `python -m jakal_search.gui_api` and stores feedback in `outputs/feedback/`.

## Deployment

Windows deployment scripts live in `deploy/`.

Install:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\run_search.ps1 -Query "graph search systems"
```

Smoke check:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\verify_bundle.ps1
```

Export a portable bundle:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\export_bundle.ps1
```

## License

This repository does not currently include a checked-in `LICENSE` file.
