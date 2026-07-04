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
uv run pytest tests/ -q --tb=short                # Run all tests (-v는 특정 실패 파고들 때만 — tests/CLAUDE.md 참고)
uv run ruff check src tests                       # Lint
```

## Key Conventions

- **Async-first:** The entire stack uses async (asyncpg, async_sessionmaker, AsyncApp). Never introduce sync DB calls.
- **SQLAlchemy 2.0 style:** Use `Mapped`, `mapped_column`, `DeclarativeBase`. No legacy 1.x patterns.
- **Embedding dimension:** Vector(768) — matches nomic-embed-text. If switching models, update `tech_item.py` and create a new Alembic migration.
- **Enum values:** Use PascalCase for all enum values (Mainstream, Alpha, Replace, Enhance, Fork, Keep, Tracking, Archived).
- **Python version:** Target >=3.10. Use `from __future__ import annotations` where needed for newer type syntax.
- **Slack handlers:** Ack within 3s, then do real work in the background. Hold the Ollama model lock across unload→query for Deep Dive so the 8B/32B swap is atomic. Asset status changes use upsert to stay concurrency-safe; every transition is logged to `track_history`. Briefings post one threaded message per item, replies stay in-thread.

## Linear 이슈 & PR 작성 관례 — 행동의 언어

이슈/PR 본문은 **행동의 언어**로 쓴다: 사용자가 앱을 만져서 참/거짓을 판정할 수 있는 문장만. 구조의 언어(파일 경로, 노드 배선, 모델명, 라이브러리 선택)는 구현자의 재량이므로 본문에 넣지 않는다. 판단 기준 두 개: (1) *이 문장을 나중에 검토 체크리스트로 그대로 쓸 수 있는가* → 본문에 쓴다. (2) *이 문장과 다르게 구현해도 의도가 충족될 수 있는가* → 본문에서 뺀다(힌트로 강등).

### 부모 이슈 본문 구조

- **불편 (Problem):** 지금 무엇이 불편한가 — 사용자 경험 기준 1~3문장.
- **성공 장면 (Scenarios):** "~하면 ~가 보인다/된다" 시나리오 2~5개. 가능하면 입력 → 기대 결과 예시 포함 (예시는 가장 압축된 의도 전달 수단).
- **실패 예시:** 지금 상태, 또는 흔히 나올 잘못된 해석 ("이건 아님").
- **완료 기준 (AC):** 성공 장면을 체크박스로. 각 항목은 구현을 몰라도 판정 가능해야 한다.
- **깨지면 안 되는 것 (Constraints):** 기존 동작 유지, 새 의존성/모델 다운로드 제한, VRAM 예산 등. 기준: "이게 깨지면 다른 게 다 돼도 의도가 깨지는가." 기술 세부가 진짜 제약일 때만 **왜**와 함께 여기 적는다 (예: "임베딩 차원 768 고정 — nomic-embed-text와 일치").
- **안 해도 되는 것 (Non-goals):** 범위 밖 명시.

구현 아이디어(영향 파일 후보, 설계 스케치)가 있으면 버리지 말고 이슈 **코멘트**에 `구현 힌트 (non-binding — 따를 의무 없음)` 라벨을 붙여 분리한다. 본문에 섞으면 나중에 구현이 힌트와 다르게 갔을 때 위반인지 재량인지 판정할 수 없게 된다.

### PR 본문 구조

PR 본문은 구현 설명서가 아니라 **검증 지도**다. 구현 요약보다 먼저:

1. **무엇이 달라지나** — 머지 시 사용자가 관찰할 행동 변화 2~4문장.
2. **AC Mapping** — 연결된 이슈의 AC **원문 그대로** × 판정(✅/⚠️/❌) × 증거(테스트·commit·수동 확인).
3. **이슈와 달라진 것 / 승인 안 된 결정** — 새 의존성·다운로드·스키마·기존 동작 변경·이슈 힌트와 다른 접근. 없으면 "없음" 명시.
4. **직접 확인하는 법** — diff 없이 판정 가능한 명령 + 기대 결과 3~5단계.

Implementation Summary는 그 뒤에 참고용으로만.

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
