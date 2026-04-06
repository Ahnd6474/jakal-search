# jakal-search

Tree-based exploratory web search engine with:

- recursive topic splitting
- embedding-based deduplication
- DBSCAN / HDBSCAN clustering
- trust-aware filtering
- pruning and stopping rules
- vector-path tracking for branch quality

## Quick Start

```bash
pip install -e .[dev]
python -m jakal_search "graph search systems"
```

For transformer embeddings:

```bash
pip install -e .[dev,models]
```

For Windows Intel GPUs via DirectML:

```bash
pip install -e .[dev,models,directml]
python -m jakal_search "graph search systems" --device directml
```

## Notes

- `sentence-transformers` is optional. If it is not installed, the engine falls back to a lightweight hashing embedder.
- On Windows with Intel Iris Xe, `torch-directml` is the practical GPU path. Official PyTorch `xpu` wheels are aimed at newer Intel Arc and Core Ultra GPUs.
- When `sentence-transformers` is active, trust scoring uses an embedding MLP head trained from trusted/suspicious prototype texts. Topic generation still uses the TF-IDF heuristic path.
- `hdbscan` is optional. If it is not installed, the engine falls back to `DBSCAN`.
- The bundled CLI uses DuckDuckGo's HTML endpoint as a simple provider.

## Output Formats

```bash
python -m jakal_search "graph search systems" --format report
python -m jakal_search "graph search systems" --format urls
python -m jakal_search "graph search systems" --format tree
python -m jakal_search "graph search systems" --format json
```

## Local Tuning

```bash
python -m jakal_search.tuning
```

This runs a local benchmark with topic-splitting cases plus adversarial trust holdouts and writes the best heuristic config to `outputs/tuning/best_config.json`.

## Trust Training

```bash
python -m jakal_search.trust_training run --embedding-device directml
```

This collects labeled trust data from official positive sources and labeled spam corpora, trains a supervised trust head on sentence-transformer embeddings, and writes the model to `outputs/models/`.

Use the trained head at runtime like this:

```bash
python -m jakal_search "graph search systems" --device directml --trust-model outputs/models/trust_head_sentence-transformers__all-MiniLM-L6-v2.pt
```
