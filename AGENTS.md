# Argos — AGENTS.md

> Full project conventions (git workflow, key conventions, constraints) live in `CLAUDE.md`
> at the repo root — this file focuses on architecture/structure for agent onboarding and
> defers to `CLAUDE.md` wherever the two would otherwise duplicate.

## Project Overview

Argos is a local-first Slack bot (plus an optional local web PWA) that automatically
tracks AI technology trends, filters hype from substance, and manages a personal
"tech asset" portfolio. It runs entirely on a MacBook Pro M1 Max 32GB with **zero
cloud cost**.

- **GitHub:** https://github.com/james20140802/argos
- **Linear:** https://linear.app/sangchu/project/argos-be0d97316a41 (team: Sangchu, prefix: SAN)

## Architecture (5 Epics)

| Epic                 | Scope                                              | Key Tech                                                  |
| -------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 1 - Local Infra      | Docker PostgreSQL + pgvector, ORM, migrations      | pgvector/pgvector:pg16, SQLAlchemy 2.0 async, Alembic     |
| 2 - Crawler          | Static (GitHub/HN/arXiv/RSS) + Dynamic (Playwright/SPA) fetchers | httpx, Playwright, readability-lxml               |
| 3 - Processing Brain | Triage → Embed → Genealogist → Digest → Save pipeline | Ollama (Qwen3-8B / 14B / 32B), LangGraph, nomic-embed-text |
| 4 - Slack Interface  | Daily/weekly briefing, Keep/Pass/Deep Dive actions, signal & succession alerts | slack_bolt AsyncApp, Socket Mode, Block Kit |
| 5 - Web (PWA)        | Local FastAPI web layer for browsing the feed/portfolio from other devices | FastAPI, uvicorn, Jinja2, HTMX, Tailscale (HTTPS transport) |

Model defaults are config-driven (`src/argos/config.py`, `OllamaConfig` /
`GenealogistConfig` / `DigestConfig`) — do not hardcode model names in prose beyond
what's listed below, since they can change per-benchmark:

- Triage: `qwen3:8b`
- Genealogist / Deep Dive: `qwen3:32b`
- Digest (longform, ARG-173): `qwen3:14b` (config-driven; may move to a benchmarked
  quantized MoE, see project memory)
- Embeddings: `nomic-embed-text` (768-dim, matches `Vector(768)` in `tech_item.py`)

## Project Structure

```
argos/
├── docker-compose.yml       # pgvector/pgvector:pg16
├── init.sql                 # CREATE EXTENSION vector, uuid-ossp
├── .env.example             # DB credentials template (dev-from-source fallback)
├── pyproject.toml           # Dependencies & tool config
├── alembic.ini
├── alembic/
│   ├── env.py               # async engine, auto-imports Base.metadata
│   └── versions/
├── src/argos/
│   ├── cli.py               # `argos` entry point — see CLI section below
│   ├── main.py              # Slack AsyncApp bootstrap (Socket Mode)
│   ├── config.py            # pydantic-settings; loads ~/.config/argos/{.env,config.toml}
│   ├── database.py          # async engine + async_sessionmaker
│   ├── scheduler.py         # launchd plist rendering/install (see Constraints below)
│   ├── models/               # SQLAlchemy 2.0 models
│   │   ├── base.py             # DeclarativeBase, UUID PK mixin, Timestamp mixin
│   │   ├── tech_item.py        # Core table — pgvector embedding, category enum, digest
│   │   ├── tech_succession.py  # Self-referential FK (predecessor/successor)
│   │   ├── user_asset.py       # Keep/Tracking/Archived status
│   │   ├── track_history.py    # Status change audit log
│   │   └── crawl_queue.py      # Staging queue for crawled-but-unprocessed items
│   ├── crawler/              # static_fetcher, dynamic_fetcher (Playwright), spa_fetcher,
│   │                          # arxiv_fetcher, rss_fetcher, add_url, pipeline, _robots, _og_image
│   ├── brain/                 # LangGraph pipeline: triage → embed → genealogist → digest → save
│   │                          # (nodes/, ollama_client.py, llm_client.py, preflight.py, weekly_report.py)
│   ├── slack/                 # app, blocks, briefing, handlers/, services/
│   ├── web/                   # FastAPI app (app.py), templates/, static/, assets/, services/
│   ├── init_wizard/           # `argos init` interactive bootstrap steps
│   └── services/              # shared cross-layer services
└── tests/
    ├── conftest.py
    ├── test_models.py       # Model unit tests (no DB required)
    ├── brain/               # triage / genealogist / digest / ollama client coverage
    ├── crawler/             # fetchers, robots, pipeline (with fixtures)
    ├── slack/               # handlers, blocks, briefing, asset transitions
    └── web/                 # route + template tests
```

## Database Schema (ERD)

- **tech_items** — id(UUID PK), title, source_url(unique), image_url, raw_content,
  summary, digest, embedding(Vector 768), category(Mainstream|Alpha), trust_score,
  published_at, briefed_at, created_at, updated_at
- **tech_succession** — id(UUID PK), predecessor_id(FK→tech_items), successor_id(FK→tech_items),
  relation_type(Replace|Enhance|Fork), reasoning
- **user_assets** — id(UUID PK), tech_id(FK→tech_items), status(Keep|Tracking|Archived), last_monitored_at
- **track_history** — id(UUID PK), user_asset_id(FK→user_assets), changed_from, changed_to, changed_at
- **crawl_queue** — staging table for freshly crawled items not yet processed by the brain
  pipeline (daily-limit throttle, ARG-93)

All FK deletions use CASCADE. All tables have UUID primary keys.

## CLI (`argos`)

Installed as a console script (`pipx install argos-scout`, or `uv run argos` from
source). Run `uv run argos --help` / `uv run argos <command> --help` for the
authoritative, current list — the table below is a quick reference and can drift.

| Command | Purpose |
| --- | --- |
| `argos run` | Full crawl → brain → save pipeline |
| `argos add <URL>...` | Manually inject URL(s) into the brain pipeline |
| `argos slack` | Start the Slack bot (Socket Mode) |
| `argos brief [--channel] [--weekly]` | Dispatch today's (or weekly) briefing to Slack |
| `argos web [--host] [--port]` | Start the local FastAPI web app (PWA) |
| `argos search <query> [--limit] [--category] [--status]` | Semantic search over `tech_items` (pgvector cosine via nomic-embed-text) |
| `argos portfolio [--category] [--sort]` | Display your Keep portfolio |
| `argos stats [--days]` | Collection-status dashboard |
| `argos backfill-images [--refetch] [--upgrade-favicons]` | Fill missing `image_url` (favicon by default; network fallback chain with flags) |
| `argos backfill-digests [--limit] [--dry-run]` | Generate longform `digest` for rows where it's NULL (LLM, slow) |
| `argos config {path,get,set,list,migrate-env}` | Read/update `~/.config/argos/config.toml` |
| `argos doctor` | Pre-flight health probes (Docker, Ollama, Python, macOS) |
| `argos init [--reconfigure <section>]` | Interactive bootstrap wizard (infra/slack/interests/schedule) |
| `argos schedule {install,uninstall,status}` | Manage the launchd jobs for `argos run` / `argos brief` |
| `argos --version` | Print installed package version |

See `CLAUDE.md`'s "Development Commands" section for copy-pasteable invocations.

## Development Commands

```bash
# Docker DB
docker compose up -d
docker compose down

# Environment
uv sync --all-extras
cp .env.example ~/.config/argos/.env && chmod 600 ~/.config/argos/.env
# Existing repo-root .env users: uv run argos config migrate-env

# Alembic migrations
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
uv run alembic downgrade -1

# Tests / lint
uv run pytest tests/ -v
uv run ruff check src tests
```

## Key Conventions

- **Async-first:** The entire stack uses async (asyncpg, async_sessionmaker, AsyncApp,
  FastAPI). Never introduce sync DB calls.
- **SQLAlchemy 2.0 style:** Use `Mapped`, `mapped_column`, `DeclarativeBase`. No legacy
  1.x patterns.
- **Embedding dimension:** `Vector(768)` — matches `nomic-embed-text`. If switching
  embedding models, update `tech_item.py` and create a new Alembic migration.
- **Enum values:** PascalCase for all enum values (Mainstream, Alpha, Replace, Enhance,
  Fork, Keep, Tracking, Archived).
- **Python version:** `>=3.10,<3.13`. Use `from __future__ import annotations` where
  needed for newer type syntax.
- **Config layering:** Secrets (Slack tokens, DB password) live in
  `~/.config/argos/.env`; behavior/tuning knobs live in `~/.config/argos/config.toml`.
  Model names, num_ctx, thresholds, etc. are config-driven — don't hardcode them in
  new code.
- **Web layer:** binds to `127.0.0.1` only; Tailscale (`tailscale serve`) is
  responsible for exposing it over HTTPS to a tailnet. Never bind `0.0.0.0`.

See `CLAUDE.md` for the full set of conventions (Slack handler ack/background-work
rules, launchd weekday convention, release/tagging process, robots allowlist, etc.) —
they are not repeated here to avoid drift between the two files.

## Git Workflow

- **Branch naming:** `feat/`, `fix/`, `docs/`, `refactor/`, `chore/` prefixes. English,
  lowercase, hyphen-separated.
- **Commits:** Atomic commits with gitmoji prefix. One logical change per commit.

## Constraints

- **Zero cloud cost.** Everything runs locally on M1 Max 32GB. No paid APIs, no cloud DB.
- **VRAM budget:** Only one LLM loaded at a time. The smaller model must be unloaded
  (`keep_alive: 0`) before loading a larger one (e.g. 8B before 32B).
- **Rate limiting:** Crawlers must use User-Agent rotation, exponential backoff, and
  respect robots.txt.
- **Robots allowlist:** see `CLAUDE.md` for the exact current allowlist and rules
  around expanding it — don't duplicate it here to avoid drift.
