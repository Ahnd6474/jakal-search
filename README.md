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

## Notes

- `sentence-transformers` is optional. If it is not installed, the engine falls back to a lightweight hashing embedder.
- `hdbscan` is optional. If it is not installed, the engine falls back to `DBSCAN`.
- The bundled CLI uses DuckDuckGo's HTML endpoint as a simple provider.
