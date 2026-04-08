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

## Desktop GUI

The desktop app uses `Tauri + React` and presents results as a blind pairwise comparison.

- The user only enters a query and picks whether result `A` or `B` is better.
- Search settings remain hidden.
- A pairwise policy learner updates internal search profiles from user preference feedback.

Start the desktop app:

```bash
cd desktop
npm install
npm run tauri dev
```

The GUI calls `python -m jakal_search.gui_api` under the hood and stores feedback state in `outputs/feedback/`.
If the desktop app cannot discover Python or the repo root automatically, set `JAKAL_SEARCH_PYTHON` and `JAKAL_SEARCH_ROOT`.

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

## Production Models

The current Windows deployment uses these trained heads:

- `outputs/models/trust_head_sentence-transformers__all-MiniLM-L6-v2.pt`
- `outputs/models/claim_falsehood_head_sentence-transformers__all-MiniLM-L6-v2.pt`
- `outputs/models/branch_head.pt`
- `outputs/models/topic_reranker_head.pt`

Run with all production heads enabled:

```bash
python -m jakal_search "graph search systems" \
  --device directml \
  --trust-model outputs/models/trust_head_sentence-transformers__all-MiniLM-L6-v2.pt \
  --claim-model outputs/models/claim_falsehood_head_sentence-transformers__all-MiniLM-L6-v2.pt \
  --branch-model outputs/models/branch_head.pt \
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

Export a portable bundle with code and model files:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\export_bundle.ps1
```
