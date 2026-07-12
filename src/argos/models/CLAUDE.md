# Argos Models — `src/argos/models/`

> Scope: SQLAlchemy ORM models (Epic 1). Loads when working under
> `src/argos/models/`. Root conventions (async-first, zero-cloud, …) still apply —
> see repo-root `CLAUDE.md`. This file adds only model-layer specifics.

## Tables (ERD)

- **tech_items** — id(UUID PK), title, source_url(unique), image_url, raw_content,
  summary, digest, embedding(`Vector(768)`), category(Mainstream|Alpha), trust_score,
  published_at, briefed_at, created/updated_at. (+ trust_rubric, corroboration_count)
- **tech_succession** — predecessor_id/successor_id (FK→tech_items),
  relation_type(Replace|Enhance|Fork), reasoning.
- **user_assets** — tech_id(FK→tech_items), status(Keep|Tracking|Archived),
  last_monitored_at.
- **track_history** — user_asset_id(FK→user_assets), changed_from, changed_to,
  changed_at — asset status-change audit log.
- **crawl_queue** — staging for crawled-but-unprocessed items (daily-limit throttle, ARG-93).

All tables have **UUID primary keys**; all FK deletions are **CASCADE**. Shared PK /
timestamp mixins live in `base.py`.

## Conventions

- **SQLAlchemy 2.0 style only.** `Mapped`, `mapped_column`, `DeclarativeBase`. No
  legacy 1.x patterns.
- **Enum values are PascalCase**: Mainstream, Alpha, Replace, Enhance, Fork, Keep,
  Tracking, Archived.
- **Embedding dimension is `Vector(768)`** — matches `nomic-embed-text`. If you switch
  embedding models, update `tech_item.py` **and** create a new Alembic migration.
- **Async-first (inherited).** Models are consumed via the async session; never add a
  sync DB call.

## Migrations

Schema changes go through Alembic — never edit the DB by hand:

```bash
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
```
