# Argos — 전체 로드맵

**Date:** 2026-05-11  
**Status:** Living Document  
**Scope:** MVP 이후 전체 프로젝트 시퀀스

---

## 전략적 방향

**Argos의 본질:** 지식 자산 관리 시스템 (Reeder/Obsidian에 가까운 경험)

**인터페이스 계층:**
- **CLI** — 모든 기능의 기반. Slack·Mac 앱 없이도 전 기능 동작 보장
- **Slack** — 알림/빠른 액션 채널 (데일리 브리핑 push, Keep/Pass/Deep Dive)
- **Mac App (SwiftUI)** — 주 인터페이스 (포트폴리오 탐색, 자연어 검색, 브리핑 히스토리)
- **iOS (미래)** — 서비스화 시점에 SwiftUI 코드베이스 확장

Slack 채팅 기반 UX는 포트폴리오 탐색·검색에 구조적 미스매치가 있다. v3 Mac 앱 전까지 CLI가 이를 보완하며, Mac 앱이 완성되면 Slack은 push 알림 전용 채널로 역할이 좁아진다.

---

## 프로젝트 시퀀스

### ✅ Argos MVP (완료)
**Epic 1–4:** Local Infra → Crawler → Brain → Slack Interface

- Docker PostgreSQL + pgvector, SQLAlchemy 2.0, Alembic
- GitHub Trending + HackerNews 크롤러 (static + dynamic)
- LangGraph 파이프라인: triage → embed → genealogist → save
- Slack 데일리 브리핑 + Keep/Pass/Deep Dive 액션

---

### 🔧 MVP Polish (진행 예정 | 2026-05 목표)
**ARG-47 ~ ARG-51**

MVP 완성 이후 체감 품질 향상. 새 기능보다 편의성·안정성 집중.

| 이슈 | 내용 |
|---|---|
| ARG-47 | 설정 분리: `.env`(시크릿) + `config.toml`(사용자 취향) |
| ARG-48 | `argos init` 위저드 (docker·alembic·ollama·launchd 일괄 설정) |
| ARG-49 | `argos config get/set/list/path` 서브커맨드 |
| ARG-50 | 관심 토픽 → triage 노드 injection |
| ARG-51 | launchd 브리핑 스케줄러 plist |

---

### 📡 Source Expansion (진행 예정 | 2026-06 목표)
**ARG-52 ~ ARG-54**

GitHub/HN에서 AI 기업 블로그 + Reddit + arXiv로 수집 소스 확대.

| 이슈 | 내용 |
|---|---|
| ARG-52 | RSS Fetcher (OpenAI/Anthropic/Google/Meta/MSR/HF 블로그 + Reddit) |
| ARG-53 | arXiv Fetcher (cs.AI/cs.LG/cs.CL, abstract only, 최근 24h) |
| ARG-54 | Category assignment: triage 노드에 Mainstream/Alpha 분류 추가 |

---

### 🔭 Track Loop (진행 예정 | 2026-06 목표)
**ARG-55 ~ ARG-57**

Keep한 기술에 대해 후속 신호·대체 기술 알림 자동화.

| 이슈 | 내용 |
|---|---|
| ARG-55 | Signal Match — 벡터 유사도 기반 후속 신호 감지 (threshold 0.85) |
| ARG-56 | Succession Alert — predecessor가 Keep-ed asset이면 즉시 알림 |
| ARG-57 | `argos brief --weekly` — Keep 목록 전체 현황 주간 요약 |

---

### 🌐 Multi-Domain (진행 예정 | 2026-07 목표)
**ARG-58 ~ ARG-62**

AI 분야를 넘어 임의 도메인(바이오테크, 핀테크 등)으로 확장. 범용성 확보.

**Phase 1: Foundation (ARG-58~60)**

| 이슈 | 내용 |
|---|---|
| ARG-58 | DB 스키마: `domains` + `tech_item_domains` + `user_assets.domain_id` |
| ARG-59 | Domain config: `config.toml`에 `[[domain]]` 블록 |
| ARG-60 | FetcherProtocol 표준화: `RawItem` + `FetcherProtocol` + registry |

**Phase 2: Integration (ARG-61~62)**

| 이슈 | 내용 |
|---|---|
| ARG-61 | 파이프라인 domain 태깅 (수집 시점부터 domain_id 흐름) |
| ARG-62 | 도메인별 채널 브리핑 + Keep domain-scoped |

---

### 💻 CLI & Interface Enhancement (Backlog | 2026-07 목표)
**ARG-63 ~ ARG-66**

CLI-first 철학 구현. Slack을 알림 채널로 역할 재정의.

| 이슈 | 내용 | 우선순위 |
|---|---|---|
| ARG-63 | `argos add <URL>` — 수동 URL 파이프라인 투입 (CLI + Slack) | High |
| ARG-64 | `argos search <query>` — 벡터 검색 기반 기술 검색 | High |
| ARG-65 | `argos portfolio` — CLI 포트폴리오 조회 | Medium |
| ARG-66 | `argos stats` — 수집 현황 통계 | Medium |

---

### 🖥️ Mac App (SwiftUI) (Backlog | 2026-Q4 목표)
**ARG-67 ~ ARG-70**

Argos의 주 인터페이스. CLI Enhancement 완료 후 시작.

| 이슈 | 내용 | 우선순위 |
|---|---|---|
| ARG-67 | SwiftUI 프로젝트 셋업 + 로컬 DB 연결 (PostgresNIO) | Medium |
| ARG-68 | 포트폴리오 뷰 — Keep-ed 기술 목록, 필터링, Untrack | Medium |
| ARG-69 | 브리핑 탐색 뷰 — 날짜별 히스토리, Keep/Pass 액션 | Medium |
| ARG-70 | 자연어 검색 UI — Spotlight 스타일, 벡터 검색 | High |

**확장:** 서비스화 시 SwiftUI 코드베이스를 iOS로 확장.

---

## 전체 타임라인

| 프로젝트 | 목표 시점 | 상태 |
|---|---|---|
| Argos MVP | 2026-05 | ✅ 완료 |
| MVP Polish | 2026-05 | 계획됨 |
| Source Expansion | 2026-06 | 계획됨 |
| Track Loop | 2026-06 | 계획됨 |
| Multi-Domain | 2026-07 | 계획됨 |
| CLI & Interface | 2026-07 | Backlog |
| Mac App (SwiftUI) | 2026-Q4 | Backlog |

---

## 상세 설계 문서

- [Argos v2 Crawler & Track Loop 설계](./2026-05-11-argos-v2-crawler-and-track-loop-design.md) — Source Expansion + Track Loop 상세 아키텍처
