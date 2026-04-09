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
- [Deployment](#deployment)
- [License](#license)

## Installation

Install the published package:

```bash
pip install jakal-search
jakal-search "graph search systems"
```

On Windows, DirectML is the practical GPU path for Intel integrated graphics:

```bash
pip install "jakal-search[directml]"
jakal-search "graph search systems" --device directml
```

For local development:

```bash
pip install -e .[dev]
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

Write the full JSON snapshot to disk:

```bash
python -m jakal_search "semiconductor AI demand" --json-out outputs/search-run.json
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

At a high level, each recursive expansion step does this:

1. Search the web with the current query.
2. Fetch the top landing pages and extract readable text.
3. Split pages into passages and rerank documents with lexical, dense, provider, freshness, and entity signals.
4. Filter low-trust or high-risk material.
5. Project document vectors through the asset-conditioned key transform.
6. Let a global fixed theme bank attend over those document keys and values.
7. Recompute scalar theme state from the gathered evidence, then propagate it through a directed asymmetric theme graph.
8. Decode parallel topic queries from the fixed themes plus the current theme state.
9. Search again with those decoded topic queries until the theme state stops moving.

The current theme layer is not just "keyword extraction." It uses a global fixed theme basis, recomputes soft evidence state from documents on every step, and uses that state to produce the next round of topic queries.

## Why the theme graph matters

The graph is there to let indirect relationships survive the search stage without recreating themes from scratch at every step.

Examples:

- AI demand -> semiconductors
- war ending -> reconstruction -> construction
- higher oil -> airlines under pressure

The engine does not hard-code those examples as labels. Instead, it keeps a fixed global theme basis, propagates scalar theme state through a directed asymmetric graph, and decodes new topic queries from the propagated state.

## Embeddings

`sentence-transformers` is a required runtime dependency. The engine always uses transformer embeddings.

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
| `json` | Full JSON run snapshot |
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
| `--max-topics` | Global topic budget |
| `--frontier-width` | How many pending topics stay in the frontier |
| `--results-per-query` | Provider results per search |
| `--fetch-top-k` | How many results get landing-page enrichment |
| `--no-page-fetch` | Skip landing-page fetching |
| `--provider-pack` | Choose `general`, `technical`, `research`, `news`, or `market_news` |
| `--stock-news` | Shortcut for market-news settings |
| `--as-of` | Strict historical cutoff |
| `--format` | Output mode |
| `--json-out` | Write full JSON snapshot to a file |
| `--device` | `auto`, `cpu`, `cuda`, `mps`, `xpu`, or `directml` |
| `--trust-model` | Trained trust head |
| `--expansion-model` | Trained expansion decision head |
| `--topic-reranker-model` | Trained topic/query reranker |
| `--claim-model` | Trained false-claim head |

## Examples

### Market-news collection for downstream modeling

```bash
python -m jakal_search "semiconductor AI demand" \
  --stock-news \
  --as-of 2026-04-08T09:00:00Z \
  --format records \
  --json-out outputs/semiconductor-run.json
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
- `jakal-search-expand`
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
  --expansion-model outputs/models/expansion_head.pt \
  --topic-reranker-model outputs/models/topic_reranker_head.pt
```

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
