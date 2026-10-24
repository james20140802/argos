"""사건 이름·요약·상세 생성 — ARG-269 / ARG-310 / ARG-308.

**여기가 사건 층에서 LLM이 들어오는 유일한 자리다.** 묶기 판단(edge_weight /
choose_event)에는 LLM을 쓰지 않는다 — 같은 입력이 항상 같은 배정을 내야
하는 결정성 요구 때문이다. 이름과 요약은 사람이 읽는 산출물이라 그 제약이
없다.

ARG-310: 모델은 ``event_summary.model``(기본 ``gemma4:26b``)이고
``get_event_summary_llm_client()``의 ``"large"`` 슬롯으로 부른다. VRAM은 한 번에
한 모델만 허용하므로 호출은 ``exclusive=True``다. 컨텍스트
(``event_summary.num_ctx``)를 넘기면 청크로 나눠 맵리듀스로 접는다
(``event_summary_input``).

ARG-308: 산출물이 "제목 + 한 단락"에서 **제목 + 짧은 요약(2~3문장) + 사건
유형 + 유형별 고정 소제목 상세**로 바뀌었다(``event_detail_schema``). 근거는
사건에 묶인 **기사 본문 전부**다 — 기사별 요약(``tech_items.summary``)과 장문
요약(``digest``)은 더 이상 입력에 넣지 않는다. 본문을 기사 수로 나눠 한
프롬프트에 다 넣고, 그래도 넘치면 맵리듀스다.

이름 검증: 기사에서 뽑힌 회사·제품 이름(라틴 문자) 중 프롬프트에 실제로 나온
상위 몇 개가 출력에 영어 원문 그대로 없으면(음역됐으면) 이름을 짚어 주며 **한 번**
다시 짓는다. JSON이 깨졌을 때도 같다. 그래도 실패하면 ``None`` — 부르는 쪽은
아무것도 쓰지 않아 이전 이름·요약이 그대로 남는다.

실패는 ``None``으로 돌린다. 부르는 쪽이 폴백(``derive_title``)을 쥐고 있고, 명명
실패가 저장이나 배정을 막으면 안 된다.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain._language import language_directive
from argos.brain.event_detail_schema import (
    EVENT_TYPE_LABELS,
    EVENT_TYPES,
    HEADINGS,
    EventDetail,
    clean_raw,
    missing_names,
    parse_event_detail,
    required_names,
    response_schema,
    scrub,
)
from argos.brain.event_summary_input import (
    BLOCK_SEPARATOR,
    MAX_FOLD_DEPTH,
    PER_DOC_FLOOR_CHARS,
    budget_chars,
    pack_chunks,
    per_doc_budget,
)
from argos.brain.llm_client import OllamaClient, get_event_summary_llm_client
from argos.brain.source_domain import registered_domain
from argos.config import settings
from argos.models.tech_event import TechEvent

logger = logging.getLogger(__name__)

MAX_EVIDENCE_CHARS = 400
"""근거 한 건의 제목이 차지할 수 있는 글자 수. 본문은 기사 수에 따라 나눈다."""

NO_BODY = "(본문 없음)"
"""본문이 비어 있는 기사 자리에 넣는 표시 — 모델이 내용을 지어내지 않게 한다."""

EventNaming = EventDetail
"""ARG-308 이전 이름. 이제 제목·요약에 사건 유형과 소제목 상세까지 담는다."""

# ARG-274 리뷰에서 고정한 마크다운 정리 규칙. 테스트가 이 이름으로 부른다.
_scrub = scrub
_clean = clean_raw


@dataclass(frozen=True)
class EvidenceDoc:
    """명명 근거 한 건 — 그 사건에 매달린 기사."""

    title: str | None
    summary: str | None = None
    """ARG-308부터 프롬프트에 넣지 않는다(기사 본문 전부가 근거). 호환용으로만 남는다."""
    body: str | None = None
    """기사 본문(ARG-310). 비어 있으면 ``(본문 없음)``으로 표시한다."""
    source: str | None = None
    """기사 URL 또는 도메인. 프롬프트에는 도메인만 쓴다 (관점이 다른 출처를 구분)."""
    names: tuple[str, ...] = field(default_factory=tuple)
    """기사에서 뽑힌 회사·제품 이름 원문(ARG-308 이름 검증용)."""


def _headings_table() -> str:
    lines = []
    for event_type in EVENT_TYPES:
        headings = " / ".join(HEADINGS[event_type])
        lines.append(f"  {event_type} ({EVENT_TYPE_LABELS[event_type]}): {headings}")
    return "\n".join(lines)


_PROMPT = """You are a technology news editor. Every article below reports the SAME event. Read ALL of them and write one structured account of the event.

Return ONE JSON object and nothing else:
{{"title": ..., "summary": ..., "event_type": ..., "sections": [{{"heading": ..., "body": ...}}]}}

Fields:
- title: one line stating WHAT HAPPENED (max 60 characters). Not a copy of any article headline, not a question.
- summary: 2-3 sentences for a feed card.
- event_type: exactly one of Launch, Research, Incident, Update, Adoption.
- sections: pick headings ONLY from the list for the chosen event_type, in that list's wording, and fill each with 2-5 sentences. Skip a heading the articles give no facts for. At least one section.
{headings}

Rules:
- Use ONLY facts in the articles. Do NOT invent numbers, features, dates or quotes. When an article body is short or missing, write only what it actually says — a short account is correct, padding is not.
- When articles take different positions or perspectives (for example the company's announcement vs. critics or independent testers), cover EVERY perspective in the sections and say whose view it is.
- Write company, product, model and repository names exactly as in the articles, in their original Latin spelling. Never translate or transliterate them.
- No markdown inside the strings.{name_note}

Articles:
{evidence}{language_reminder}"""

_NAME_NOTE = (
    "\n- These names MUST appear verbatim (original spelling, not transliterated): {names}."
)

_MEMO_PROMPT = """You are preparing fact notes for a news editor. Below are articles that all report the SAME event. Write compact notes of the concrete facts they contain: who did what, when, numbers, product and company names.

Rules:
- Plain text, at most 15 short lines, no markdown, no preamble.
- Start each line with the article number it comes from, like [2]. When articles take different positions or perspectives, keep each article's stance as its own line.
- Keep proper nouns, product names and code identifiers exactly as written in the articles (do not translate or transliterate them).
- Use ONLY facts present below. Do NOT invent details.

Articles:
{evidence}"""


def _header(index: int, doc: EvidenceDoc) -> str:
    title = scrub(doc.title or "")[:MAX_EVIDENCE_CHARS]
    lines = [f"[{index}] {title}".rstrip()]
    domain = registered_domain(doc.source) if doc.source else None
    if domain or doc.source:
        lines.append(f"출처: {domain or doc.source}")
    return "\n".join(lines)


def _format_block(index: int, doc: EvidenceDoc, body_chars: int) -> str:
    body = (doc.body or "").strip()[:body_chars]
    return f"{_header(index, doc)}\n본문: {body or NO_BODY}"


def _format_evidence(docs: Sequence[EvidenceDoc], budget: int, doc_max_chars: int) -> list[str]:
    """기사마다 한 블록. 본문은 남은 예산을 기사 수로 나눈 만큼(``per_doc_budget``)."""
    headers = sum(len(_header(i, d)) + 16 for i, d in enumerate(docs, start=1))
    per_doc = per_doc_budget(
        budget=max(budget - headers, 0),
        n_docs=len(docs),
        floor=PER_DOC_FLOOR_CHARS,
        cap=doc_max_chars,
    )
    return [_format_block(i, d, per_doc) for i, d in enumerate(docs, start=1)]


class _CallLog:
    """한 사건 명명 안의 LLM 호출 비용 집계 (ARG-310 관측)."""

    def __init__(self, num_ctx: int) -> None:
        self.num_ctx = num_ctx
        self.calls = 0
        self.prompt_tokens = 0
        self.started = time.monotonic()

    def add(self, stats: dict) -> None:
        self.calls += 1
        tokens = int(stats.get("prompt_eval_count") or 0)
        self.prompt_tokens += tokens
        if tokens and tokens >= self.num_ctx:
            # Ollama는 넘친 앞부분을 조용히 버린다 -- 품질 버그가 숨지 않게 알린다.
            logger.warning(
                "event summary prompt used %d tokens >= num_ctx %d -- input may be truncated",
                tokens,
                self.num_ctx,
            )


async def generate_event_naming(
    docs: Sequence[EvidenceDoc],
    *,
    client: OllamaClient | None = None,
    keep_alive: str | int = "5m",
    label: str | None = None,
) -> EventDetail | None:
    """근거 기사들로 사건 제목·요약·유형·소제목 상세를 짓는다. 실패하면 ``None``.

    ``None``이 되는 경우: 근거가 하나도 없음 / LLM 호출 실패 / 두 번 다 출력이
    검증(``parse_event_detail``)을 못 넘거나 필수 이름이 빠짐 / 맵리듀스가
    ``MAX_FOLD_DEPTH``단계 안에 안 접힘.

    ``client``를 넘기면 재사용한다 -- 백필이 배치 전체에서 모델을 한 번만
    적재하기 위해서다. ``label``은 로그에 찍을 사건 식별자다.
    """
    evidence_docs = [doc for doc in docs if (doc.title or doc.body)]
    if not evidence_docs:
        return None

    cfg = settings.user.event_summary
    if client is None:
        client = get_event_summary_llm_client()
    language = settings.user.slack.summary_language or "English"
    reminder = language_directive(language)
    headings = _headings_table()
    schema = response_schema()

    # 이름 짚기(재시도) 몫까지 템플릿 길이에 넣어 둔다.
    template_chars = (
        max(len(_PROMPT) + len(headings) + 400, len(_MEMO_PROMPT)) + len(reminder)
    )
    budget = budget_chars(
        num_ctx=cfg.num_ctx,
        output_reserve_tok=cfg.output_reserve_tok,
        chars_per_tok=cfg.chars_per_tok,
        template_chars=template_chars,
    )
    log = _CallLog(cfg.num_ctx)

    async def _ask(prompt: str, *, structured: bool) -> str:
        stats: dict = {}
        raw = await client.query(
            "large",
            prompt,
            keep_alive=keep_alive,
            timeout=cfg.timeout,
            num_ctx=cfg.num_ctx,
            think=False,
            temperature=cfg.temperature,
            exclusive=True,
            stats_out=stats,
            format=schema if structured else None,
        )
        log.add(stats)
        return raw

    used_docs = evidence_docs[: cfg.max_docs]
    blocks = _format_evidence(used_docs, budget, cfg.doc_max_chars)
    chunk_count = 1
    detail: EventDetail | None = None
    attempts = 0
    try:
        items = blocks
        depth = 0
        while True:
            chunks = pack_chunks(items, budget)
            if len(chunks) <= 1:
                break
            if depth >= MAX_FOLD_DEPTH:
                logger.warning(
                    "generate_event_naming: evidence still over budget after %d folds", depth
                )
                return None
            if depth == 0:
                chunk_count = len(chunks)
            memo_cap = max(budget // 4, 1)  # 메모가 청크만큼 부풀어 접기가 안 끝나는 일을 막는다
            memos: list[str] = []
            for chunk in chunks:
                raw = await _ask(
                    _MEMO_PROMPT.format(evidence=BLOCK_SEPARATOR.join(chunk)), structured=False
                )
                memo = clean_raw(raw or "")[:memo_cap]
                if not memo:
                    logger.warning("generate_event_naming: empty fact memo")
                    return None
                memos.append(f"[notes {len(memos) + 1}]\n{memo}")
            items = memos
            depth += 1
        evidence = BLOCK_SEPARATOR.join(chunks[0]) if chunks else ""

        required = required_names(
            [doc.names for doc in used_docs],
            evidence,
            titles=[doc.title for doc in used_docs],
        )
        for attempt in range(2):
            attempts = attempt + 1
            note = ""
            if attempt and required:
                note = _NAME_NOTE.format(names=", ".join(required))
            raw = await _ask(
                _PROMPT.format(
                    headings=headings,
                    name_note=note,
                    evidence=evidence,
                    language_reminder=reminder,
                ),
                structured=True,
            )
            candidate = parse_event_detail(raw or "")
            if candidate is None:
                logger.warning(
                    "generate_event_naming %s: attempt %d output failed validation",
                    label or "-",
                    attempts,
                )
                continue
            missing = missing_names(candidate.text(), required)
            if missing:
                logger.warning(
                    "generate_event_naming %s: attempt %d dropped names %s",
                    label or "-",
                    attempts,
                    missing,
                )
                continue
            detail = candidate
            break
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_event_naming failed: %r", exc)
        return None

    logger.info(
        "event naming %s: docs=%d chunks=%d calls=%d attempts=%d ok=%s prompt_tokens=%d "
        "seconds=%.1f",
        label or "-",
        len(blocks),
        chunk_count,
        log.calls,
        attempts,
        detail is not None,
        log.prompt_tokens,
        time.monotonic() - log.started,
    )
    return detail


async def apply_event_naming(
    session: AsyncSession,
    event_id: uuid.UUID,
    docs: Sequence[EvidenceDoc],
    *,
    client: OllamaClient | None = None,
    keep_alive: str | int = "5m",
    expected_updated_at: datetime | None = None,
) -> bool:
    """사건에 이름·요약·유형·소제목 상세를 지어 붙이고 ``naming_stale``을 내린다.

    생성이 실패하면 **아무것도 쓰지 않고** ``False``를 돌린다 — 이전 이름·요약·
    상세가 그대로 남고, 사건은 ``naming_stale``(또는 ``sections IS NULL``)인 채로
    남아 뒤의 재요약 패스가 줍는다. 이 함수는 예외를 삼키지 않는다(생성 실패는
    ``generate_event_naming``이 이미 ``None``으로 접어 준다). 세이브포인트는 부르는
    쪽이 건다.

    ARG-274: ``expected_updated_at``을 주면 그 값이 아직 그대로일 때만 쓴다.
    근거를 스냅샷으로 뜬 뒤 LLM이 도는 동안 온라인 파이프라인이 같은 사건에
    문서를 하나 더 매달 수 있는데, 그러면 ``link_document_to_event``가
    ``naming_stale``을 다시 세운다. 가드가 걸리면 아무것도 쓰지 않고 ``False``를
    돌린다 — 다음 패스가 새 근거까지 포함해 다시 짓는다.

    ``updated_at``은 ``TimestampMixin``의 ``onupdate``라 Core ``update()``에도
    붙는다. 온라인 경로는 방금 만든 사건을 자기 트랜잭션 안에서 명명하므로
    끼어들 여지가 없어 가드를 넘기지 않는다(기본값 ``None``).
    """
    naming = await generate_event_naming(
        docs, client=client, keep_alive=keep_alive, label=str(event_id)
    )
    if naming is None:
        return False

    stmt = update(TechEvent).where(TechEvent.id == event_id)
    if expected_updated_at is not None:
        stmt = stmt.where(TechEvent.updated_at == expected_updated_at)
    result = await session.execute(
        stmt.values(
            title=naming.title,
            summary=naming.summary,
            event_type=naming.event_type,
            sections=naming.sections_json(),
            naming_stale=False,
        )
    )
    if expected_updated_at is not None and result.rowcount == 0:
        logger.info(
            "apply_event_naming: %s changed while naming ran — leaving it stale",
            event_id,
        )
        return False
    return True
