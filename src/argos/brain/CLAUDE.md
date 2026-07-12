# Argos Brain — `src/argos/brain/`

> Scope: the LangGraph processing pipeline (Epic 3). Loads when working under
> `src/argos/brain/`. Root conventions (async-first, zero-cloud, VRAM budget, …)
> still apply — see repo-root `CLAUDE.md`. This file adds only brain-layer specifics.

## Pipeline

`triage → embed → genealogist → digest → save` (LangGraph, nodes in `nodes/`).
Supporting modules: `ollama_client.py`, `llm_client.py`, `trust.py`,
`corroboration.py`, `preflight.py`, `graph_state.py`, `weekly_report.py`.

## Conventions

- **Model names are config-driven — never hardcode them.** Defaults live in
  `src/argos/config.py` (`OllamaConfig` / `GenealogistConfig` / `DigestConfig`) and
  may change per-benchmark. Current defaults:
  - Triage: `qwen3:8b`
  - Genealogist / Deep Dive: `qwen3:32b`
  - Digest (longform): `qwen3:14b`
  - Embeddings: `nomic-embed-text` (768-dim, matches `Vector(768)`)
- **VRAM budget: one LLM loaded at a time.** The smaller model must be unloaded
  (`keep_alive: 0`) before loading a larger one (8B before 32B). For Deep Dive, hold
  the Ollama model lock across unload→query so the 8B↔32B swap is **atomic**.
- **Zero cloud cost (inherited).** No paid APIs — everything runs locally via Ollama.
- **Async-first (inherited).** Pipeline nodes and the save step use the async session.
