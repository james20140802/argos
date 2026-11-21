"""사건 심층 분석 생성 — ARG-335.

사건에 묶인 **기사 전부**를 읽고 다섯 소제목(핵심 기술·접근 / 성숙도 / 생태계와의
관계 / 위험·한계 / 추적할 만한가)으로 나눠 쓴 분석을 만든다. 순수 생성 단계다 —
DB에 아무것도 쓰지 않는다(저장·표시는 부르는 쪽 몫).

구조는 ``event_naming.generate_event_naming``을 본으로 삼는다: 입력 예산
(``event_summary_input``) 안에 기사 블록을 넣고, 넘치면 묶음별 사실 메모로 접고,
구조화 출력(JSON schema)을 최대 2회 시도하며, 영어 이름 보존 검사
(``event_detail_schema.required_names``)를 같은 규칙으로 건다. 사건 요약 동작은
건드리지 않는다.

실패는 전부 :class:`DeepAnalysisError`로 접는다. 예상 못 한 예외는 삼키지 않는다.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain._language import language_directive
from argos.brain.event_detail_schema import (
    clean_raw,
    missing_names,
    required_names,
    scrub,
)
from argos.brain.event_naming import MAX_EVIDENCE_CHARS, NO_BODY
from argos.brain.event_summary_input import (
    BLOCK_SEPARATOR,
    MAX_FOLD_DEPTH,
    budget_chars,
    pack_chunks,
    per_doc_budget,
)
from argos.brain.llm_client import OllamaClient, get_deep_analysis_llm_client
from argos.brain.ollama_client import OllamaInfraError
from argos.brain.source_domain import registered_domain
from argos.config import settings

logger = logging.getLogger(__name__)

DEEP_ANALYSIS_HEADINGS: tuple[str, ...] = (
    "핵심 기술·접근",
    "성숙도",
    "생태계와의 관계",
    "위험·한계",
    "추적할 만한가",
)
SECTION_KEYS: tuple[str, ...] = ("core", "maturity", "ecosystem", "risks", "verdict")
"""``DEEP_ANALYSIS_HEADINGS``와 같은 순서. 모델은 키만 쓰고 코드가 소제목을 붙인다."""

INSUFFICIENT_EVIDENCE = "근거 부족"
DEEP_PER_DOC_FLOOR_CHARS = 500
"""기사당 본문 몫이 이 아래로 내려가면 묶음별 사실 메모로 접는다."""


class DeepAnalysisFailure(str, Enum):
    NoEvidence = "NoEvidence"  # 사건에 기사가 없음
    ModelUnavailable = "ModelUnavailable"  # 모델 응답 없음(연결 실패·5xx)
    Timeout = "Timeout"  # 시간 초과
    BadFormat = "BadFormat"  # 형식 불량(소제목 누락·빈 칸·JSON 아님·필수 이름 누락)
    TooLarge = "TooLarge"  # MAX_FOLD_DEPTH 안에 안 접힘


FAILURE_MESSAGES: dict[DeepAnalysisFailure, str] = {
    DeepAnalysisFailure.NoEvidence: "이 사건에 묶인 기사가 없습니다",
    DeepAnalysisFailure.ModelUnavailable: "모델이 응답하지 않습니다",
    DeepAnalysisFailure.Timeout: "시간이 초과됐습니다",
    DeepAnalysisFailure.BadFormat: "모델이 정해진 형식으로 답하지 않았습니다",
    DeepAnalysisFailure.TooLarge: "기사가 너무 많아 한 번에 읽지 못했습니다",
}


class DeepAnalysisError(Exception):
    reason: DeepAnalysisFailure

    def __init__(self, reason: DeepAnalysisFailure, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}" if detail else reason.value)


@dataclass(frozen=True)
class DeepAnalysisDoc:
    id: uuid.UUID
    title: str | None
    body: str | None = None
    source: str | None = None
    published_at: datetime | None = None
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeepAnalysisSection:
    heading: str
    body: str


@dataclass(frozen=True)
class DeepAnalysis:
    sections: tuple[DeepAnalysisSection, ...]
    """항상 5개, ``DEEP_ANALYSIS_HEADINGS`` 순서."""
    doc_ids: tuple[uuid.UUID, ...]
    """근거로 쓴 기사 id 전부(입력 순서)."""
    doc_count: int
    model: str

    def sections_json(self) -> list[dict[str, str]]:
        """``tech_events.sections``와 같은 모양."""
        return [{"heading": s.heading, "body": s.body} for s in self.sections]


_PROMPT = """당신은 기술 동향 분석가다. 아래 기사들은 모두 같은 사건을 다룬다. 기사를 전부 읽고 이 사건을 다섯 관점으로 분석하라.

JSON 객체 하나만 돌려준다. 다른 글은 쓰지 않는다:
{{"core": ..., "maturity": ..., "ecosystem": ..., "risks": ..., "verdict": ...}}

각 키의 내용 (각각 2~5문장):
- core: 핵심 기술·접근 — 무엇을 어떤 방식으로 하는가.
- maturity: 성숙도 — 실험 단계인지, 실사용·상용 단계인지.
- ecosystem: 생태계와의 관계 — 기존 기술·경쟁 제품·표준과 어떻게 맞물리는가.
- risks: 위험·한계 — 기사에 드러난 약점, 제약, 비판.
- verdict: 추적할 만한가 — 이 사건을 계속 지켜볼 가치가 있는지 이유와 함께.

규칙:
- 기사에 없는 내용은 지어내지 않는다. 숫자·기능·날짜·인용을 만들지 않는다. 어떤 소제목에 쓸 근거가 기사에 없으면 그 칸에 `{insufficient}`이라고 쓰고 무엇이 없는지 한 문장으로 적는다.
- 기사마다 입장이 다르면(발표 주체 vs 비판·독립 검증) 누구의 시각인지 밝혀 둘 다 반영한다.
- 회사·제품·모델 이름은 번역·음차하지 말고 영어 원문 그대로 쓴다.
- 문자열 안에 마크다운을 쓰지 않는다.{name_note}

기사들:
{evidence}{language_reminder}"""

_NAME_NOTE = "\n- 다음 이름은 원문 철자 그대로 반드시 나와야 한다(음차 금지): {names}."

_MEMO_PROMPT = """아래는 같은 사건을 다룬 기사들이다. 분석가가 쓸 사실 메모를 만들어라: 누가 무엇을 했는지, 시점, 숫자, 제품·회사 이름, 한계·비판, 성숙도(실험/상용)와 경쟁·연동 관계.

규칙:
- 일반 텍스트, 최대 15줄, 마크다운·머리말 없음.
- 각 줄은 출처 기사 번호로 시작한다. 예: [2]. 기사마다 입장이 다르면 그 입장을 별도 줄로 남긴다.
- 고유명사·제품명·코드 식별자는 기사에 적힌 그대로 둔다(번역·음차 금지).
- 아래 기사에 있는 사실만 쓴다. 지어내지 않는다.

기사들:
{evidence}"""

_DATE_FMT = "%Y-%m-%d"
_BODY_LABEL = "\n본문: "


def _header(index: int, doc: DeepAnalysisDoc) -> str:
    title = scrub(doc.title or "")[:MAX_EVIDENCE_CHARS]
    lines = [f"[{index}] {title}".rstrip()]
    domain = registered_domain(doc.source) if doc.source else None
    if domain or doc.source:
        lines.append(f"출처: {domain or doc.source}")
    if doc.published_at is not None:
        lines.append(f"날짜: {doc.published_at.strftime(_DATE_FMT)}")
    return "\n".join(lines)


def _format_evidence(
    docs: Sequence[DeepAnalysisDoc], budget: int, doc_max_chars: int
) -> list[str]:
    """기사마다 한 블록. 머리글은 항상 넣고 본문은 남은 예산을 기사 수로 나눈 만큼."""
    headers = [_header(i, d) for i, d in enumerate(docs, start=1)]
    overhead = sum(len(h) + len(_BODY_LABEL) + len(BLOCK_SEPARATOR) for h in headers)
    per_doc = per_doc_budget(
        budget=max(budget - overhead, 0),
        n_docs=len(docs),
        floor=DEEP_PER_DOC_FLOOR_CHARS,
        cap=doc_max_chars,
    )
    blocks = []
    for header, doc in zip(headers, docs):
        body = (doc.body or "").strip()[:per_doc]
        blocks.append(f"{header}{_BODY_LABEL}{body or NO_BODY}")
    return blocks


def _response_schema() -> dict:
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in SECTION_KEYS},
        "required": list(SECTION_KEYS),
    }


def _load_object(raw: str) -> dict | None:
    cleaned = clean_raw(raw or "")
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _parse_sections(raw: str) -> tuple[DeepAnalysisSection, ...] | None:
    payload = _load_object(raw)
    if payload is None:
        return None
    sections = []
    for key, heading in zip(SECTION_KEYS, DEEP_ANALYSIS_HEADINGS):
        value = payload.get(key)
        body = scrub(value) if isinstance(value, str) else ""
        if not body:
            return None
        sections.append(DeepAnalysisSection(heading, body))
    return tuple(sections)


def _classify(exc: OllamaInfraError) -> DeepAnalysisError:
    if isinstance(exc.__cause__, httpx.TimeoutException):
        return DeepAnalysisError(DeepAnalysisFailure.Timeout, str(exc))
    return DeepAnalysisError(DeepAnalysisFailure.ModelUnavailable, str(exc))


class _CallLog:
    """한 사건 분석 안의 LLM 호출 비용 집계."""

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
            logger.warning(
                "deep analysis prompt used %d tokens >= num_ctx %d -- input may be truncated",
                tokens,
                self.num_ctx,
            )


async def generate_deep_analysis(
    docs: Sequence[DeepAnalysisDoc],
    *,
    client: OllamaClient | None = None,
    keep_alive: str | int = "5m",
    label: str | None = None,
) -> DeepAnalysis:
    """근거 기사 전부로 다섯 소제목 분석을 짓는다. 실패는 전부 ``DeepAnalysisError``."""
    used_docs = [d for d in docs if (d.title or "").strip() or (d.body or "").strip()]
    if not used_docs:
        raise DeepAnalysisError(DeepAnalysisFailure.NoEvidence)

    event_cfg = settings.user.event_summary
    resolved = settings.user.deep_analysis_resolved()
    if client is None:
        client = get_deep_analysis_llm_client()
    language = settings.user.slack.summary_language or "English"
    reminder = language_directive(language)
    schema = _response_schema()

    # 이름 짚기(재시도) 몫까지 템플릿 길이에 넣어 둔다.
    template_chars = max(len(_PROMPT) + len(reminder) + 400, len(_MEMO_PROMPT))
    budget = budget_chars(
        num_ctx=resolved.num_ctx,
        output_reserve_tok=resolved.output_reserve_tok,
        chars_per_tok=event_cfg.chars_per_tok,
        template_chars=template_chars,
    )
    log = _CallLog(resolved.num_ctx)

    async def _ask(prompt: str, *, structured: bool) -> str:
        stats: dict = {}
        try:
            raw = await client.query(
                "large",
                prompt,
                keep_alive=keep_alive,
                timeout=resolved.timeout,
                num_ctx=resolved.num_ctx,
                think=False,
                temperature=event_cfg.temperature,
                exclusive=True,
                stats_out=stats,
                format=schema if structured else None,
            )
        except OllamaInfraError as exc:
            raise _classify(exc) from exc
        log.add(stats)
        return raw

    blocks = _format_evidence(used_docs, budget, event_cfg.doc_max_chars)
    chunk_count = 1
    attempts = 0
    sections: tuple[DeepAnalysisSection, ...] | None = None
    error: DeepAnalysisError | None = None
    try:
        items = blocks
        depth = 0
        while True:
            chunks = pack_chunks(items, budget)
            if len(chunks) <= 1:
                break
            if depth >= MAX_FOLD_DEPTH:
                raise DeepAnalysisError(
                    DeepAnalysisFailure.TooLarge, f"still over budget after {depth} folds"
                )
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
                    raise DeepAnalysisError(DeepAnalysisFailure.BadFormat, "empty fact memo")
                memos.append(f"[notes {len(memos) + 1}]\n{memo}")
            items = memos
            depth += 1
        evidence = BLOCK_SEPARATOR.join(chunks[0]) if chunks else ""

        required = required_names(
            [d.names for d in used_docs],
            evidence,
            titles=[d.title for d in used_docs],
        )
        for attempt in range(2):
            attempts = attempt + 1
            note = _NAME_NOTE.format(names=", ".join(required)) if attempt and required else ""
            raw = await _ask(
                _PROMPT.format(
                    insufficient=INSUFFICIENT_EVIDENCE,
                    name_note=note,
                    evidence=evidence,
                    language_reminder=reminder,
                ),
                structured=True,
            )
            candidate = _parse_sections(raw or "")
            if candidate is None:
                logger.warning(
                    "deep analysis %s: attempt %d output failed validation", label or "-", attempts
                )
                continue
            missing = missing_names("\n".join(s.body for s in candidate), required)
            if missing:
                logger.warning(
                    "deep analysis %s: attempt %d dropped names %s", label or "-", attempts, missing
                )
                continue
            sections = candidate
            break
        if sections is None:
            error = DeepAnalysisError(
                DeepAnalysisFailure.BadFormat, f"no valid output after {attempts} attempts"
            )
    except DeepAnalysisError as exc:
        error = exc

    logger.info(
        "deep analysis %s: docs=%d chunks=%d calls=%d attempts=%d ok=%s prompt_tokens=%d "
        "seconds=%.1f",
        label or "-",
        len(blocks),
        chunk_count,
        log.calls,
        attempts,
        error is None,
        log.prompt_tokens,
        time.monotonic() - log.started,
    )
    if error is not None:
        raise error
    assert sections is not None
    doc_ids = tuple(d.id for d in used_docs)
    return DeepAnalysis(
        sections=sections, doc_ids=doc_ids, doc_count=len(doc_ids), model=resolved.model
    )


_EVENT_DOCS_SQL = text(
    """
    SELECT i.id AS doc_id,
           i.title AS doc_title,
           i.source_url AS doc_source,
           i.published_at AS doc_published_at,
           LEFT(i.raw_content, :doc_max_chars) AS doc_body,
           ARRAY(
               SELECT en.name
               FROM document_entities de
               JOIN entities en ON en.id = de.entity_id
               WHERE de.tech_item_id = i.id
               ORDER BY en.name
           ) AS doc_names
    FROM event_documents ed
    JOIN tech_items i ON i.id = ed.tech_item_id
    WHERE ed.event_id = :event_id
    ORDER BY i.published_at ASC NULLS LAST, i.id
    """
)


async def fetch_event_docs(
    session: AsyncSession, event_id: uuid.UUID
) -> list[DeepAnalysisDoc]:
    """그 사건에 묶인 기사 전부(개수 상한 없음). 사건이 없으면 빈 리스트.

    병합 툼스톤은 따라가지 않는다 — 그 사건 id에 묶인 것만 읽는다.
    """
    rows = (
        await session.execute(
            _EVENT_DOCS_SQL,
            {"event_id": event_id, "doc_max_chars": settings.user.event_summary.doc_max_chars},
        )
    ).all()
    return [
        DeepAnalysisDoc(
            id=row.doc_id,
            title=row.doc_title,
            body=row.doc_body,
            source=row.doc_source,
            published_at=row.doc_published_at,
            names=tuple(row.doc_names or ()),
        )
        for row in rows
    ]


async def analyze_event(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    client: OllamaClient | None = None,
) -> DeepAnalysis:
    """``fetch_event_docs`` → ``generate_deep_analysis``. 아무것도 저장하지 않는다."""
    docs = await fetch_event_docs(session, event_id)
    return await generate_deep_analysis(docs, client=client, label=str(event_id))
