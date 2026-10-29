# Argos — CLAUDE.md

## Project Overview

Argos is a local-first Slack bot that automatically tracks AI technology trends, filters hype from substance, and manages a personal "tech asset" portfolio. It runs entirely on a MacBook Pro M1 Max 32GB with **zero cloud cost**.

- **GitHub:** https://github.com/james20140802/argos
- **Linear:** https://linear.app/sangchu/project/argos-be0d97316a41 (workspace: Sangchu, team: Argos, prefix: ARG)

## Architecture (5 Epics)

| Epic                 | Scope                                              | Key Tech                                                  |
| -------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 1 - Local Infra      | Docker PostgreSQL + pgvector, ORM, migrations      | pgvector/pgvector:pg16, SQLAlchemy 2.0 async, Alembic     |
| 2 - Crawler          | Static (GitHub/HN/arXiv/RSS) + Dynamic (Playwright/SPA) fetchers | httpx, Playwright, readability-lxml               |
| 3 - Processing Brain | Triage → Embed → Genealogist → Assign event → Save pipeline; event summary + per-type detail sections from every article body | Ollama (Qwen3-8B / 32B, gemma4:26b for event summaries), LangGraph, nomic-embed-text |
| 4 - Slack Interface  | Daily/weekly briefing, Keep/Pass/Deep Dive actions, signal & succession alerts | slack_bolt AsyncApp, Socket Mode, Block Kit |
| 5 - Web (PWA)        | Local FastAPI web layer for browsing the feed/portfolio from other devices | FastAPI, uvicorn, Jinja2, HTMX, Tailscale (HTTPS transport) |

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

- **tech_items** — id(UUID PK), title, source_url(unique), image_url, raw_content,
  summary, digest(legacy, no longer generated — ARG-308), embedding(Vector 768), category(Mainstream|Alpha), trust_score,
  published_at, briefed_at, created_at, updated_at
- **tech_succession** — id(UUID PK), predecessor_id(FK→tech_items), successor_id(FK→tech_items), relation_type(Replace|Enhance|Fork), reasoning
- **user_assets** — id(UUID PK), tech_id(FK→tech_items), status(Keep|Tracking|Archived), last_monitored_at
- **track_history** — id(UUID PK), user_asset_id(FK→user_assets), changed_from, changed_to, changed_at
- **tech_events** — id(UUID PK), title, summary(2~3 sentences), event_type(Launch|Research|Incident|Update|Adoption, NULL),
  sections(JSONB `[{heading, body}]`, NULL = not yet re-summarized — ARG-308), occurred_at, naming_stale, merged_into_id(self-FK, RESTRICT)
- **push_subscriptions** — id(UUID PK), endpoint(unique), p256dh, auth, label(기기 이름), origin(VAPID sub), last_success_at, created_at, updated_at
  (아침 브리핑 웹 푸시를 켠 기기, ARG-321). 푸시 서비스가 404/410이면 행 삭제
- **push_dispatches** — id(UUID PK), sent_at, event_count, title, devices_ok/failed/expired
  (브리핑 발송 회차, 전부 실패한 회차도 devices_ok=0으로 기록, ARG-321). 다음 회차 "새 사건" 창은 한 기기라도 받은 마지막 sent_at부터
- **crawl_queue** — staging table for freshly crawled items not yet processed by the brain
  pipeline (daily-limit throttle, ARG-93)
- **role_branches** — id(UUID PK), parent_id(FK→role_branches, CASCADE, NULL=위 갈래), name(unique), description, position, confirmed_at
  (확정된 역할 갈래 트리, ARG-311). 행이 있으면 사람이 확정한 것 — 쓰기 경로는 `argos taxonomy apply` 하나뿐
- **technology_profiles** — id(UUID PK), entity_id(FK→entities, CASCADE, unique; 대표 이름), verdict(Technology|NotTechnology|Insufficient),
  not_tech_reason, kind, description(한국어 한 줄), role_en, described_doc_count, described_at, model (ARG-313; entities.kind는 불변)
- **role_assignments** — id(UUID PK), entity_id(FK→entities, CASCADE, unique), branch_id(FK→role_branches, CASCADE, NULL=미분류),
  evidence(JSONB 배정 근거), unclassified_reason, source(Auto|Manual — 자동 배치는 Manual 행 미변경), assigned_at, run_id (ARG-313)

All FK deletions use CASCADE, with exactly one exception: the merge-tombstone
self-FK `tech_events.merged_into_id` uses **RESTRICT**. A merge is a tombstone,
not a delete — the absorbed event keeps its row and only points at the survivor,
so CASCADE there would delete the very events the pointer exists to preserve.
Do not "fix" it to CASCADE for consistency. The model relationship also needs
`passive_deletes="all"`, or SQLAlchemy nulls the pointer before the DELETE and
the RESTRICT never fires (see `src/argos/models/tech_event.py`).

All tables have UUID primary keys.

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
uv run argos brief [--channel CID] [--weekly] [--push-only]  # Morning web push to subscribed devices, then Slack (or weekly; --push-only skips Slack)
uv run argos web [--host] [--port]                # Start local FastAPI web app (PWA), binds 127.0.0.1 only
uv run argos web token [--rotate] [--url BASE]   # Print/rotate the web login token (terminal only; 1 login per device)
uv run argos search <query> [--limit] [--category] [--status]  # Semantic search over tech_items (pgvector)
uv run argos portfolio [--category] [--sort]      # Display your Keep portfolio
uv run argos stats [--days]                       # Collection-status dashboard
uv run argos backfill-images [--refetch] [--upgrade-favicons]  # Fill missing image_url (favicon by default)
uv run argos backfill-digests                     # Retired (ARG-308): prints a notice pointing to resummarize-events
uv run argos resummarize-events [--limit] [--dry-run] [--yes] [--sample N]  # Re-summarize events (summary + per-type sections) from every article body (preview → confirm → LLM write)
uv run argos backfill-entities [--dry-run] [--yes] [--sample N]  # Rewrite stored entity names for existing articles (preview → confirm → write)
uv run argos alias-names [--dry-run] [--yes] [--sample N] [--show NAME]  # Merge different names of one technology into an alias layer (preview → confirm → write; stored names untouched)
uv run argos correct-events [--from] [--to] [--dry-run]  # Apply event merge/split corrections for a period (--dry-run: read-only preview)
uv run argos config {path,get,set,list,migrate-env}  # Read/update ~/.config/argos/config.toml
uv run argos taxonomy draft [--seed] [--keep-top] [--limit] [--file]  # 역할 갈래 초안 → 파일(DB 미기록). --seed는 LLM 없음
uv run argos taxonomy {rename,merge,split,delete} ...  # 초안 파일 위 편집 (이름·합치기·나누기·지우기)
uv run argos taxonomy apply [--dry-run] [--yes]        # 차이 미리보기 → y/N → role_branches 확정 (DB 쓰기는 이것뿐)
uv run argos taxonomy {show,export}                     # 확정본 보기 / 확정본을 초안 파일로 내보내 다시 고치기
uv run argos taxonomy assign [--all] [--name N]... [--names-from F] [--limit N] [--dry-run] [--yes] [--sample N]  # 새 기술 → 아래 갈래 자동 배정 (LLM 계획 → 미리보기 → y/N → 같은 계획 기록). --all은 소급
uv run argos taxonomy why NAME                          # 기술 하나의 설명·갈래·배정 근거
uv run argos taxonomy eval [--truth F] [--tree confirmed|spike] [--assignments F]  # 정답 묶음 대비 쌍 일치도 (읽기 전용; spike는 LLM 사용)

# launchd scheduler (macOS) — see src/argos/scheduler.py
uv run argos schedule install                     # Render + bootstrap all plists from config
uv run argos schedule status                      # Show loaded/not-loaded for all labels
uv run argos schedule uninstall                   # Bootout all plists (idempotent; web/correct/classify plists are also deleted)
# Plists land at:  ~/Library/LaunchAgents/com.argos.{run,brief,correct,classify}.plist
# Logs land at:    ~/Library/Logs/argos/{run,brief,correct,classify}.log
# Times configured via:  briefing.time, briefing.weekdays, run.time, correction.time (daily `correct-events`; must differ from run/brief times),
#                        taxonomy.assign_time (daily `taxonomy assign --yes`, default 04:30; must differ from all of the above; taxonomy.assign_enabled=false to turn off)

# Tests
uv run pytest tests/ -q --tb=short                # Run all tests (-v는 특정 실패 파고들 때만 — tests/CLAUDE.md 참고)
uv run ruff check src tests                       # Lint
```

## Key Conventions

- **Async-first:** The entire stack uses async (asyncpg, async_sessionmaker, AsyncApp). Never introduce sync DB calls.
- **SQLAlchemy 2.0 style:** Use `Mapped`, `mapped_column`, `DeclarativeBase`. No legacy 1.x patterns.
- **Embedding dimension:** Vector(768) — matches nomic-embed-text. If switching models, update `tech_item.py` and create a new Alembic migration.
- **Enum values:** Use PascalCase for all enum values (Mainstream, Alpha, Replace, Enhance, Fork, Keep, Tracking, Archived).
- **Python version:** Target >=3.10,<3.13. Use `from __future__ import annotations` where needed for newer type syntax.
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
