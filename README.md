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

Argos는 로컬 머신에서 완전히 실행되므로, 아래 네 가지가 미리 설치되어 있어야 합니다.

| 항목 | 버전 | 비고 |
|------|------|------|
| Docker / [Colima](https://github.com/abiosoft/colima) | 최신 | PostgreSQL + pgvector 컨테이너 실행용 |
| [Ollama](https://ollama.com) | 최신 | Qwen3-8B / 32B 로컬 추론용 |
| Slack 워크스페이스 | — | 봇 설치 권한 필요 |
| Python 3.10–3.12 | >=3.10, <3.13 | 3.13은 아직 지원하지 않음 |

## Install via pipx

[pipx](https://pipx.pypa.io/)를 사용하면 독립된 가상환경에 Argos를 설치할 수 있습니다.

```bash
pipx install argos-scout
```

> **기본 Python이 지원 범위 밖인 경우** (3.13 이상 또는 3.9 이하):
>
> ```bash
> pipx install --python python3.12 argos-scout
> ```

### Bootstrap

설치 후 인터랙티브 위저드로 환경을 설정합니다 (Docker 컨테이너 기동, Alembic 마이그레이션,
Ollama 모델 다운로드, Slack 토큰 검증, launchd 스케줄 등록까지 한 번에 진행합니다).

```bash
argos init
```

특정 섹션만 다시 설정하고 싶다면:

```bash
argos init --reconfigure slack       # infra / slack / interests / schedule
```

위저드는 idempotent 하게 동작합니다 — 기존 `~/.config/argos/.env` / `~/.config/argos/config.toml` 값을
다시 디폴트로 보여주고, 사용자가 바꾼 값만 atomic하게 다시 씁니다. 시크릿 값은
재표시 시 항상 마스킹되며 (`xoxb-***` / `***`), `.env` 파일은 항상 `chmod 600` 으로
잠깁니다.

> **기존 repo-root `.env` 사용자:** `argos config migrate-env` 를 실행하면
> 기존 `.env`를 `~/.config/argos/.env`로 이동하고 원본은 `.env.bak`으로 백업합니다.

### Verify

설치와 환경이 올바른지 확인합니다.

```bash
argos doctor       # Docker / Ollama / Python / macOS 프리플라이트 체크
argos --version    # 설치된 패키지 버전 출력
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

`~/.config/argos/.env` 에 아래 세 값을 채웁니다 (`.env.example` 참고).

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

> **자동 스케줄링:** `argos init` 위저드가 macOS `launchd` 잡을 자동으로 등록합니다 (`com.argos.run`, `com.argos.brief`). 직접 관리하려면 `argos schedule` 서브커맨드를 쓰세요. 플리스트는 `~/Library/LaunchAgents/`에, 로그는 `~/Library/Logs/argos/{run,brief}.log`에 남습니다.
>
> ```bash
> uv run argos schedule install      # config.toml 기준으로 두 잡을 등록 + 부트스트랩
> uv run argos schedule status       # 두 라벨의 로드 여부 확인
> uv run argos schedule uninstall    # 두 잡을 모두 해제 (이미 없어도 에러 없음)
> ```
>
> 스케줄 시각은 `~/.config/argos/config.toml`의 `briefing.time` / `briefing.weekdays` / `run.time`로 조정합니다 (아래 `argos config` 참고).

### 5. 일반적인 워크플로우

```bash
# 1) 새 데이터 수집 + 분석 (Ollama 필요)
uv run argos run

# 2) 결과를 Slack 으로 발송
uv run argos brief

# 3) 봇 데몬을 띄워두면 Keep/Pass/Deep Dive 버튼이 동작
uv run argos slack
```

## Configuration (`argos config`)

런타임 설정은 `~/.config/argos/config.toml`에 저장됩니다 (위저드가 생성). 시크릿(Slack 토큰, DB 비밀번호 등)은 `~/.config/argos/.env`에 두고, 동작 옵션만 config.toml에서 관리합니다. CLI로 안전하게 읽고 쓸 수 있습니다.

```bash
uv run argos config path             # 사용 중인 config.toml 경로 출력
uv run argos config list             # 전체 키 출력 (시크릿은 마스킹됨)
uv run argos config get briefing.time
uv run argos config set briefing.time 09:00
uv run argos config set briefing.weekdays "mon,tue,wed,thu,fri"
uv run argos config set briefing.limit_per_category 5
```

종료 코드: `0` 성공 · `1` I/O 등 일반 오류 · `2` 알 수 없는 키 · `3` 검증 실패 · `4` 시크릿 거부(`.env`에서 직접 관리).

설정을 바꾼 뒤 스케줄 시각/요일이 영향받는다면 `uv run argos schedule install`을 다시 돌려 플리스트를 갱신하세요.

## Testing

```bash
# 전체 테스트 (Ollama 불필요, 모두 mocked)
uv run pytest tests/ -v

# Brain 노드만
uv run pytest tests/brain/ -v
```

## Contributing / Dev from source

소스에서 직접 개발하거나 기여하려면:

```bash
# 1. 저장소 클론 + 의존성 설치 (.venv 자동 생성)
git clone https://github.com/james20140802/argos.git
cd argos
uv sync --all-extras

# 2. 환경 파일 생성
mkdir -p ~/.config/argos
cp .env.example ~/.config/argos/.env
chmod 600 ~/.config/argos/.env

# 3. 비밀번호/토큰을 채워 넣은 뒤 인프라 기동
docker compose up -d
uv run alembic upgrade head

# 4. 위저드로 나머지 설정
uv run argos init
```

CI나 비-TTY 환경에서는 `ARGOS_INIT_NONINTERACTIVE=1` (또는
`--non-interactive`) 로 모든 디폴트를 조용히 채택할 수 있습니다.

```bash
# 테스트 실행
uv run pytest tests/ -v

# 린트
uv run ruff check src tests
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
