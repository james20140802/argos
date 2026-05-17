# Argos — CLAUDE.md

## Project Overview

Argos is a local-first Slack bot that automatically tracks AI technology trends, filters hype from substance, and manages a personal "tech asset" portfolio. It runs entirely on a MacBook Pro M1 Max 32GB with **zero cloud cost**.

- **GitHub:** https://github.com/james20140802/argos
- **Linear:** https://linear.app/sangchu/project/argos-be0d97316a41 (team: Sangchu, prefix: SAN)

## Architecture (4 Epics)

| Epic                 | Scope                                              | Key Tech                                                  |
| -------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 1 - Local Infra      | Docker PostgreSQL + pgvector, ORM, migrations      | pgvector/pgvector:pg16, SQLAlchemy 2.0 async, Alembic     |
| 2 - Crawler          | Static (GitHub/HN) + Dynamic (Playwright) fetchers | httpx, Playwright, readability-lxml                       |
| 3 - Processing Brain | Triage → Embed → Genealogist → Save pipeline       | Ollama (Qwen3-8B / 32B), LangGraph, nomic-embed-text      |
| 4 - Slack Interface  | Daily briefing, Keep/Pass/Deep Dive actions        | slack_bolt AsyncApp, Socket Mode, Block Kit               |

## Project Structure

```
argos/
├── docker-compose.yml       # pgvector/pgvector:pg16
├── init.sql                 # CREATE EXTENSION vector, uuid-ossp
├── .env.example             # DB credentials template
├── pyproject.toml           # Dependencies & tool config
├── alembic.ini
├── alembic/
│   ├── env.py               # async engine, auto-imports Base.metadata
│   └── versions/
├── src/argos/
│   ├── cli.py               # `argos` entry point — run / slack / brief subcommands
│   ├── main.py              # Slack AsyncApp bootstrap (Socket Mode)
│   ├── config.py            # pydantic-settings, loads .env
│   ├── database.py          # async engine + async_sessionmaker
│   ├── models/              # SQLAlchemy 2.0 models (tech_item, tech_succession, user_asset, track_history)
│   ├── crawler/             # static_fetcher, dynamic_fetcher (Playwright), pipeline, _robots, user_agents
│   ├── brain/               # LangGraph pipeline: triage → embed → genealogist → save (+ ollama_client)
│   └── slack/               # app, blocks, briefing, handlers/, services/
└── tests/
    ├── conftest.py
    ├── test_models.py       # Model unit tests (no DB required)
    ├── brain/               # triage / genealogist / ollama client coverage
    ├── crawler/             # fetchers, robots, pipeline (with fixtures)
    └── slack/               # handlers, blocks, briefing, asset transitions
```

## Database Schema (ERD)

- **tech_items** — id(UUID PK), title, source_url(unique), raw_content, embedding(Vector 768), category(Mainstream|Alpha), trust_score, created_at, updated_at
- **tech_succession** — id(UUID PK), predecessor_id(FK→tech_items), successor_id(FK→tech_items), relation_type(Replace|Enhance|Fork), reasoning
- **user_assets** — id(UUID PK), tech_id(FK→tech_items), status(Keep|Tracking|Archived), last_monitored_at
- **track_history** — id(UUID PK), user_asset_id(FK→user_assets), changed_from, changed_to, changed_at

All FK deletions use CASCADE. All tables have UUID primary keys.

## Development Commands

```bash
# Docker DB
docker compose up -d                              # Start PostgreSQL + pgvector
docker compose down                               # Stop

# Environment (uv manages .venv automatically)
uv sync --all-extras                              # Create .venv and install runtime + dev deps
cp .env.example ~/.config/argos/.env && chmod 600 ~/.config/argos/.env  # Create XDG env file
# Existing repo-root .env users: uv run argos config migrate-env

# Alembic migrations
uv run alembic revision --autogenerate -m "description"  # Generate migration
uv run alembic upgrade head                              # Apply migrations
uv run alembic downgrade -1                              # Rollback one step

# Argos CLI (operator entry points)
uv run argos --version                            # Print installed package version
uv run argos doctor                               # Pre-flight health check (Docker / Ollama / Python / macOS)
uv run argos init                                 # Interactive 6-step bootstrap wizard (installs Playwright Chromium automatically)
uv run argos init --reconfigure slack             # Re-run one section: infra/slack/interests/schedule
uv run argos run [--url URL]...                   # Crawl → brain → save pipeline
uv run argos add <URL> [URL ...]                  # Manually inject URL(s) into the brain pipeline
uv run argos slack                                # Start Slack bot in Socket Mode
uv run argos brief [--channel CID]                # Dispatch today's briefing

# launchd scheduler (macOS) — see src/argos/scheduler.py
uv run argos schedule install                     # Render + bootstrap both plists from config
uv run argos schedule status                      # Show loaded/not-loaded for both labels
uv run argos schedule uninstall                   # Bootout both plists (idempotent)
# Plists land at:  ~/Library/LaunchAgents/com.argos.{run,brief}.plist
# Logs land at:    ~/Library/Logs/argos/{run,brief}.log
# Times configured via:  briefing.time, briefing.weekdays, run.time

# Tests
uv run pytest tests/ -v                           # Run all tests
uv run ruff check src tests                       # Lint
```

## Key Conventions

- **Async-first:** The entire stack uses async (asyncpg, async_sessionmaker, AsyncApp). Never introduce sync DB calls.
- **SQLAlchemy 2.0 style:** Use `Mapped`, `mapped_column`, `DeclarativeBase`. No legacy 1.x patterns.
- **Embedding dimension:** Vector(768) — matches nomic-embed-text. If switching models, update `tech_item.py` and create a new Alembic migration.
- **Enum values:** Use PascalCase for all enum values (Mainstream, Alpha, Replace, Enhance, Fork, Keep, Tracking, Archived).
- **Python version:** Target >=3.10. Use `from __future__ import annotations` where needed for newer type syntax.
- **Slack handlers:** Ack within 3s, then do real work in the background. Hold the Ollama model lock across unload→query for Deep Dive so the 8B/32B swap is atomic. Asset status changes use upsert to stay concurrency-safe; every transition is logged to `track_history`. Briefings post one threaded message per item, replies stay in-thread.

## Git Workflow

- **Branch naming:** `feat/`, `fix/`, `docs/`, `refactor/`, `chore/` prefixes. English, lowercase, hyphen-separated.
- **Commits:** Atomic commits with gitmoji prefix. One logical change per commit.

### Releases

The CD workflow (`.github/workflows/release.yml`) publishes to PyPI automatically when a `v*.*.*` tag is pushed. Before tagging:

1. Bump `project.version` in `pyproject.toml` in the commit you intend to tag.
2. Create an **annotated** tag — the tag message becomes the GitHub Release body:
   ```bash
   git tag -a vMAJOR.MINOR.PATCH -m "Release notes / changelog here"
   ```
3. Push the tag:
   ```bash
   git push --tags
   ```

The workflow will fail fast if the tag version (`vX.Y.Z` → `X.Y.Z`) does not match `project.version` in `pyproject.toml`. Always use annotated tags (`git tag -a`); lightweight tags produce an empty release body.

## Constraints

- **Zero cloud cost.** Everything runs locally on M1 Max 32GB. No paid APIs, no cloud DB.
- **VRAM budget:** Only one LLM loaded at a time. 8B model must be unloaded (keep_alive: 0) before loading 32B.
- **Rate limiting:** Crawlers must use User-Agent rotation, exponential backoff, and respect robots.txt.
- **Robots allowlist:** `_robots.py` carves out a tiny set of vendor-published public-API hosts (currently `hacker-news.firebaseio.com`) whose generic robots.txt would falsely block documented public endpoints. Do not expand without a documented public-API contract.
- **launchd Weekday convention:** `scheduler._weekday_to_launchd` maps Sun=0..Sat=6 — **NOT** the ISO Mon=1..Sun=7 convention. The full table is unit-tested at `tests/test_scheduler.py::test_weekday_to_launchd_full_table`. A full 7-day list collapses to a single `StartCalendarInterval` dict with no `Weekday` key; a subset expands to a list of dicts each carrying a `Weekday` int.
