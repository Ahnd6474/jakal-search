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
- `hdbscan` is optional. If it is not installed, the engine falls back to `DBSCAN`.
- The bundled CLI uses DuckDuckGo's HTML endpoint as a simple provider.

## Output Formats

```bash
python -m jakal_search "graph search systems" --format report
python -m jakal_search "graph search systems" --format urls
python -m jakal_search "graph search systems" --format tree
python -m jakal_search "graph search systems" --format json
```
