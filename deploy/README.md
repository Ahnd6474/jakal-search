# Deployment

This directory contains the deployment entrypoints for the current Windows + Intel GPU setup.

## Included runtime models

- `trust_head_sentence-transformers__paraphrase-MiniLM-L3-v2.pt`
- `claim_falsehood_head_sentence-transformers__paraphrase-MiniLM-L3-v2.pt`
- `branch_head.pt`
- `topic_reranker_head.pt`

## Install on a target machine

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
```

## Run the search service from the repo root

```powershell
powershell -ExecutionPolicy Bypass -File deploy\run_search.ps1 -Query "graph search systems"
```

## Smoke check

```powershell
powershell -ExecutionPolicy Bypass -File deploy\verify_bundle.ps1
```

## Export a portable deployment bundle

```powershell
powershell -ExecutionPolicy Bypass -File deploy\export_bundle.ps1
```

The bundle is written to `dist\jakal-search-deploy` by default and includes code, scripts, and the current trained model files.
