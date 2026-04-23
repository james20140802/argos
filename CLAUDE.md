# Argos — CLAUDE.md

## Project Overview

Argos (Omni-Lens: Tech Scout) is a local-first Slack bot that automatically tracks AI technology trends, filters hype from substance, and manages a personal "tech asset" portfolio. It runs entirely on a MacBook Pro M1 Max 32GB with **zero cloud cost**.

- **GitHub:** https://github.com/james20140802/argos
- **Linear:** https://linear.app/sangchu/project/argos-be0d97316a41 (team: Sangchu, prefix: SAN)

## Architecture (4 Epics)

| Epic                 | Scope                                              | Key Tech                                                  |
| -------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 1 - Local Infra      | Docker PostgreSQL + pgvector, ORM, migrations      | pgvector/pgvector:pg16, SQLAlchemy 2.0 async, Alembic     |
| 2 - Crawler          | Static (GitHub/HN) + Dynamic (Playwright) fetchers | httpx, Playwright, readability-lxml                       |
| 3 - Processing Brain | Triage → Embed → Genealogist → Save pipeline       | Ollama (Llama-3-8B / 70B-Q4), LangGraph, nomic-embed-text |
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
│   ├── config.py            # pydantic-settings, loads .env
│   ├── database.py          # async engine + async_sessionmaker
│   └── models/
│       ├── base.py          # DeclarativeBase, UUID PK mixin, Timestamp mixin
│       ├── tech_item.py     # Core table — pgvector embedding, category enum
│       ├── tech_succession.py  # Self-referential FK (predecessor/successor)
│       ├── user_asset.py    # Keep/Tracking/Archived status
│       └── track_history.py # Status change audit log
└── tests/
    ├── conftest.py
    └── test_models.py       # 35 unit tests (no DB required)
```

## Database Schema (ERD)

- **tech_items** — id(UUID PK), title, source_url(unique), raw_content, embedding(Vector 1536), category(Mainstream|Alpha), trust_score, created_at, updated_at
- **tech_succession** — id(UUID PK), predecessor_id(FK→tech_items), successor_id(FK→tech_items), relation_type(Replace|Enhance|Fork), reasoning
- **user_assets** — id(UUID PK), tech_id(FK→tech_items), status(Keep|Tracking|Archived), last_monitored_at
- **track_history** — id(UUID PK), user_asset_id(FK→user_assets), changed_from, changed_to, changed_at

All FK deletions use CASCADE. All tables have UUID primary keys.

## Development Commands

```bash
# Docker DB
docker-compose up -d                              # Start PostgreSQL + pgvector
docker-compose down                               # Stop

# Alembic migrations
alembic revision --autogenerate -m "description"  # Generate migration
alembic upgrade head                              # Apply migrations
alembic downgrade -1                              # Rollback one step

# Tests
pip install -e ".[dev]"                           # Install with dev deps
pytest tests/ -v                                  # Run all tests

# Environment
cp .env.example .env                              # Create local env file
```

## Key Conventions

- **Async-first:** The entire stack uses async (asyncpg, async_sessionmaker, AsyncApp). Never introduce sync DB calls.
- **SQLAlchemy 2.0 style:** Use `Mapped`, `mapped_column`, `DeclarativeBase`. No legacy 1.x patterns.
- **Embedding dimension:** Vector(1536) — matches nomic-embed-text. If switching models, update `tech_item.py` and create a new Alembic migration.
- **Enum values:** Use PascalCase for all enum values (Mainstream, Alpha, Replace, Enhance, Fork, Keep, Tracking, Archived).
- **Python version:** Target >=3.10. Use `from __future__ import annotations` where needed for newer type syntax.

## Git Workflow

- **Branch naming:** `feat/`, `fix/`, `docs/`, `refactor/`, `chore/` prefixes. English, lowercase, hyphen-separated.
- **Commits:** Atomic commits with gitmoji prefix. One logical change per commit.

## Constraints

- **Zero cloud cost.** Everything runs locally on M1 Max 32GB. No paid APIs, no cloud DB.
- **VRAM budget:** Only one LLM loaded at a time. 8B model must be unloaded (keep_alive: 0) before loading 70B.
- **Rate limiting:** Crawlers must use User-Agent rotation, exponential backoff, and respect robots.txt.
