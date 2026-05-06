# Argos

AI 기술 동향을 자동 추적하고, 노이즈를 걸러 개인 기술 포트폴리오로 관리하는 로컬 Slack 봇.
M1 Max 32GB에서 완전히 로컬로 동작하며 클라우드 비용이 없습니다.

## Architecture

```
Crawler (GitHub Trending / HN / Playwright)
    ↓
Processing Brain (LangGraph)
    Triage → Embed → Genealogist → Save
    ↓
PostgreSQL + pgvector (Docker)
    ↓
Slack Interface (Daily briefing + Keep/Pass/Deep Dive)
```

## Prerequisites

- Docker
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (Python 패키지 매니저)
- [Ollama](https://ollama.com) (Processing Brain 실행 시 필요)

## Setup

```bash
# 1. 의존성 설치 (.venv 자동 생성)
uv sync --all-extras

# 2. 환경 변수 설정
cp .env.example .env

# 3. PostgreSQL + pgvector 시작
docker compose up -d

# 4. DB 마이그레이션 적용
uv run alembic upgrade head
```

## Running

### 전체 파이프라인 (크롤링 → 분석 → 저장)

Ollama 실행 및 모델 준비가 필요합니다:

```bash
ollama pull qwen3:8b
ollama pull qwen3:32b
ollama pull nomic-embed-text
```

CLI로 실행:

```bash
uv run argos run
```

동적 URL을 추가로 처리하려면 `--url`을 반복해서 전달:

```bash
uv run argos run --url https://example.com/article --url https://example.com/another
```

상세 로그를 보려면 `-v` / `--verbose`:

```bash
uv run argos run -v
```

### Brain 파이프라인만 단독 실행

```python
import asyncio
from argos.database import AsyncSessionLocal
from argos.brain.pipeline import run_brain_pipeline

async def main():
    async with AsyncSessionLocal() as session:
        state = await run_brain_pipeline(
            raw_text="...",
            source_url="https://example.com",
            session=session,
        )
        await session.commit()
        print(state)

asyncio.run(main())
```

## Slack Bot

Argos의 사용자 인터페이스는 Slack 봇입니다 (Socket Mode 기반, 인바운드 포트 불필요).
DB에 쌓인 그날의 기술 신호를 Block Kit 카드로 발송하고, 사용자가 **Keep / Pass / Deep Dive** 버튼으로 자산을 관리합니다.

### 1. Slack 앱 생성

1. <https://api.slack.com/apps> 에서 **Create New App → From scratch** 로 새 앱을 만듭니다 (워크스페이스 선택).
2. 좌측 메뉴 **Socket Mode** 진입 → 토글 **Enable Socket Mode** ON.
   - 그 자리에서 발급되는 **App-Level Token** (`xapp-…`) 의 스코프에 `connections:write` 가 포함되어 있어야 합니다. → `SLACK_APP_TOKEN`.
3. **OAuth & Permissions → Bot Token Scopes** 에 다음 스코프를 모두 추가합니다.

   | Scope | 용도 |
   | --- | --- |
   | `chat:write` | 브리핑 메시지 발송 |
   | `chat:write.public` | (선택) 봇 미초대 채널에도 발송 가능 |
   | `commands` | (선택) 슬래시 커맨드 확장 시 |

4. **Event Subscriptions** 진입 → **Enable Events** ON. Socket Mode이므로 Request URL은 비워둬도 됩니다. **Subscribe to bot events** 에서 `app_mention` 정도만 추가해도 무방합니다 (현재 코드는 인터랙티브 액션만 사용하므로 필수 아님).
5. **Interactivity & Shortcuts** 진입 → **Interactivity** ON. Socket Mode에서는 Request URL을 입력하지 않아도 됩니다. `action_keep` / `action_pass` / `action_deep_dive` 버튼이 이 채널을 통해 전달됩니다.
6. **Install App → Install to Workspace** 로 설치하고, 발급된 **Bot User OAuth Token** (`xoxb-…`) 을 복사합니다. → `SLACK_BOT_TOKEN`.
7. 발송할 채널 ID를 확보합니다 (Slack 채널 우클릭 → **View channel details** → 하단의 `Cxxxxxxxxxx`). 비공개 채널이면 봇을 채널에 초대 (`/invite @argos`) 해야 합니다. → `SLACK_CHANNEL_ID`.

### 2. 환경변수 설정

`.env` 에 아래 세 값을 채웁니다 (`.env.example` 참고).

```dotenv
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_CHANNEL_ID=C0123456789
```

### 3. 봇 실행 — Socket Mode 데몬

별도의 웹서버 없이 한 줄로 켭니다.

```bash
uv run argos slack
```

이 프로세스는 Slack 게이트웨이에 WebSocket을 유지하면서 Keep / Pass / Deep Dive 버튼 이벤트를 수신·처리합니다.
**3초 ACK 룰**을 지키기 위해 Deep Dive 핸들러는 즉시 응답을 보낸 뒤 70B 분석은 백그라운드 태스크에서 수행합니다.

### 4. 일일 브리핑 발송

오늘 수집·분석된 항목을 Mainstream / Alpha 카테고리별 카드로 묶어 채널에 발송합니다.

```bash
# 기본 채널 (SLACK_CHANNEL_ID) 로 발송
uv run argos brief

# 다른 채널로 임시 발송
uv run argos brief --channel C0987654321
```

당일 처리된 항목이 없으면 자동으로 발송을 건너뜁니다 (빈 메시지 방지).

> **자동 스케줄링 (선택):** 매일 정해진 시각 발송을 원한다면 macOS `launchd` 또는 `cron`을 활용하세요. 별도 클라우드 의존성을 만들지 않기 위해 Argos 자체에는 스케줄러를 두지 않았습니다.
>
> ```cron
> # 매일 오전 9시(KST) 발송 — crontab -e
> 0 9 * * * cd /path/to/argos && /usr/local/bin/uv run argos brief >> argos.log 2>&1
> ```

### 5. 일반적인 워크플로우

```bash
# 1) 새 데이터 수집 + 분석 (Ollama 필요)
uv run argos run

# 2) 결과를 Slack 으로 발송
uv run argos brief

# 3) 봇 데몬을 띄워두면 Keep/Pass/Deep Dive 버튼이 동작
uv run argos slack
```

## Testing

```bash
# 전체 테스트 (Ollama 불필요, 모두 mocked)
uv run pytest tests/ -v

# Brain 노드만
uv run pytest tests/brain/ -v
```

## Database

```bash
# 마이그레이션 생성
uv run alembic revision --autogenerate -m "description"

# 마이그레이션 적용
uv run alembic upgrade head

# 롤백
uv run alembic downgrade -1

# DB 종료
docker compose down
```
