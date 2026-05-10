# Argos v2 — Crawler Expansion & Track Loop Design

**Date:** 2026-05-11  
**Status:** Approved  
**Scope:** Two parallel epics — Crawler Source Expansion + Track Loop

---

## Background

The MVP (Epics 1–4) is complete. The current system crawls GitHub Trending and HackerNews, runs items through a LangGraph brain pipeline (triage → embed → genealogist → save), and posts daily briefings to Slack with Keep/Pass/Deep Dive actions.

Three pain points identified after MVP use:

1. **Track 루프 없음** — Keep을 눌러도 해당 기술에 대한 후속 신호나 succession 알림이 전혀 없음
2. **크롤링 출처 부족** — GitHub/HN만으로는 AI 기업 공식 발표, 논문, 커뮤니티 신호를 놓침
3. **UX** — 후순위 (이번 스코프 제외)

---

## Epic A: Crawler Source Expansion

### Goal

하루 수집 범위를 GitHub/HN에서 AI 기업 블로그 + 커뮤니티 + 논문으로 확대한다.

### New Sources

| 소스 | 방식 | 카테고리 힌트 | 비고 |
|---|---|---|---|
| OpenAI Blog | RSS | Mainstream | |
| Anthropic Blog | RSS | Mainstream | |
| Google DeepMind Blog | RSS | Mainstream | |
| Meta AI Blog | RSS | Mainstream | |
| Microsoft Research Blog | RSS | Mainstream | |
| Hugging Face Blog | RSS | Mainstream | |
| r/MachineLearning | Reddit RSS | Alpha | OAuth 불필요 |
| r/LocalLLaMA | Reddit RSS | Alpha | OAuth 불필요 |
| arXiv cs.AI / cs.LG / cs.CL | arXiv API | Alpha | abstract only, 최근 24h |

### Architecture

```
crawler/
├── static_fetcher.py      (기존 — GitHub Trending, HackerNews)
├── dynamic_fetcher.py     (기존 — Playwright)
├── rss_fetcher.py         (신규)
├── arxiv_fetcher.py       (신규)
└── pipeline.py            (기존 — 확장)
```

**rss_fetcher.py**
- `feedparser` 라이브러리로 RSS 파싱
- 소스 목록은 config(`UserConfig`)에서 관리 — 추후 추가/제거 용이
- 각 피드 항목: `title + description/summary → raw_content`, `link → source_url`
- 기존 `_robots.py` + URL 중복 체크 흐름 그대로 적용
- `source_category: CategoryType` 힌트를 함께 반환

**arxiv_fetcher.py**
- arXiv Atom API (`http://export.arxiv.org/api/query`) 사용
- 쿼리: `cat:cs.AI OR cat:cs.LG OR cat:cs.CL`, `submittedDate` 기준 최근 24시간
- `title + abstract → raw_content` (full text 제외)
- `arxiv.org/abs/{id} → source_url`
- `source_category = Alpha`

**Pipeline 통합**
- `crawler/pipeline.py`에서 기존 fetcher들과 동일하게 호출
- fetcher가 반환한 `source_category` 힌트를 `BrainState`에 추가
- Triage 노드가 힌트를 참고해 최종 category 결정 (기존 is_valid/trust_score와 동일한 패턴)

### Category Assignment (현재 gap 해결)

현재 triage 노드가 category를 결정하지 않는 문제를 이번에 함께 수정:
- triage 프롬프트에 `source_category` 힌트와 함께 "Mainstream or Alpha" 판단 추가
- `_TriageResult`에 `category: CategoryType` 필드 추가
- `save_node`에서 category 저장

---

## Epic B: Track Loop

### Goal

Keep한 기술에 대해 후속 신호와 succession 알림을 자동으로 받는다.

### Three Components

#### 1. Signal Match (후속 신호 알림)

- `argos run` 파이프라인 완료 후 실행
- 새로 저장된 `TechItem`들의 embedding과 `status=Keep`인 `user_assets` → `tech_items`의 embedding을 pgvector cosine similarity로 비교
- 유사도 > 0.85 → "관련 신호" 감지
- 감지된 신호를 묶어 Slack 브리핑 채널에 전송

#### 2. Succession Alert (대체 기술 알림)

- `argos run` 완료 후 genealogist가 생성한 `tech_succession` 레코드를 스캔
- `predecessor_id`가 Keep-ed asset의 `tech_id`와 일치하면 즉시 알림
- 알림 형식: "⚠️ [기존 기술]을 대체하는 [신기술]이 등장했습니다"

#### 3. Weekly Report

- `argos brief --weekly` 커맨드 (또는 launchd 별도 스케줄)
- Keep 목록 전체 현황: 보유 기술 수, 지난 7일간 관련 신호 수, succession 발생 여부
- 각 Keep-ed 기술별 `last_monitored_at` 업데이트

### Slack Delivery

- 매일 `argos run` 종료 시, signal/succession 결과가 있으면 브리핑 채널에 별도 메시지 전송
- 형식:

```
🔭 Track 업데이트 — 2026-05-11

• [LangGraph 0.3] → Keep한 LangGraph와 유사한 새 신호 (trust=0.82)
• ⚠️ Keep한 [LlamaIndex]를 대체하는 [LightRAG]가 등장 (genealogist 판단)
```

### Code Structure

```
slack/services/
├── asset_transition.py      (기존)
├── briefing_query.py        (기존)
└── track_check.py           (신규)
    ├── match_signals()      — 벡터 유사도 기반 신호 매칭
    ├── check_succession()   — succession 레코드 스캔
    └── post_track_update()  — Slack 전송
```

- `track_history` 테이블 활용 (이미 존재) — 알림 이력 저장
- `user_assets.last_monitored_at` 업데이트

---

## Sequencing

두 에픽은 독립적으로 병행 개발 가능:

| 순서 | 작업 |
|---|---|
| 1 | `rss_fetcher.py` + 소스 목록 config화 |
| 2 | `arxiv_fetcher.py` |
| 3 | category assignment gap 수정 (triage 노드) |
| 4 | `track_check.py` — signal match |
| 5 | `track_check.py` — succession alert |
| 6 | `argos brief --weekly` |
| 7 | Pipeline 통합 테스트 |

---

---

## MVP 다듬기 (병행 처리)

두 에픽과 별개로 현재 MVP의 체감 품질을 높이는 개선 작업.

### SAN-37: Keep 목록 조회 (포트폴리오)

Slack slash command(`/argos portfolio` 또는 `@argos portfolio`)로 Keep-ed asset 목록 조회.
- Block Kit 카드: 기술명, URL, Keep 날짜, last_monitored_at
- 각 항목에 "Untrack(Archive)" 버튼
- `briefing_query.py`에 `fetch_user_portfolio()` 추가

### SAN-38: `argos run` 실행 결과 요약

파이프라인 완료 후 콘솔 요약 출력:
```
✅ argos run 완료 (2026-05-11 09:00)
  수집: 87개  →  유효: 34개  →  저장(신규): 21개
  소스별: GitHub 12 / HN 8 / RSS 6 / arXiv 4
  소요 시간: 4m 32s
```

### SAN-39: Genealogist cold start 처리

DB 아이템 수 < N(기본 50)이면 genealogist 노드 스킵, 32B 모델 로드 낭비 방지.
- `UserConfig`에 `genealogist.min_db_items` 추가

### SAN-40: `limit_per_category` config화

`briefing_query.py`의 하드코딩 `limit_per_category=5`를 `UserConfig`(`briefing.limit_per_category`, 기본 10)로 이동.
SAN-33(category assignment) 완료 후 Mainstream/Alpha 균형 검증 필요.

---

## Out of Scope

- UX 개선 (브리핑 카드 레이아웃, TOP N 큐레이션) — 별도 epic
- Reddit 공식 API (OAuth) — RSS로 충분
- Full text 논문 fetch — abstract로 충분
- Human-Signal Filter (커밋 로그 가중치) — 미래 epic
