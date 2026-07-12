# Argos Slack — `src/argos/slack/`

> Scope: the Slack interface (Epic 4). Loads when working under `src/argos/slack/`.
> Root conventions still apply — see repo-root `CLAUDE.md`. This file adds only
> Slack-layer specifics.

## Layout

`slack_bolt` **AsyncApp** over **Socket Mode**, rendering **Block Kit**. Modules:
`app.py`, `blocks.py`, `briefing.py`, `handlers/`, `services/`.

## Conventions

- **Ack within 3s, then do real work in the background.** Slack requires a fast ack;
  offload the actual processing (LLM calls, DB writes) to a background task.
- **Deep Dive holds the Ollama model lock across unload→query** so the 8B↔32B swap is
  atomic (see `brain/` VRAM budget).
- **Asset status changes use upsert** to stay concurrency-safe, and **every transition
  is logged to `track_history`** (Keep / Tracking / Archived).
- **Briefings post one threaded message per item; replies stay in-thread.**
- **Async-first (inherited).** The whole Slack stack is async.
