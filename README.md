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
