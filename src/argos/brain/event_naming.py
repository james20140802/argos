"""사건 이름·요약 생성 — ARG-269.

**여기가 사건 층에서 LLM이 들어오는 유일한 자리다.** 묶기 판단(edge_weight /
choose_event)에는 LLM을 쓰지 않는다 — 같은 입력이 항상 같은 배정을 내야
하는 결정성 요구 때문이다. 이름과 요약은 사람이 읽는 산출물이라 그 제약이
없다.

모델은 8B(``get_llm_client()``의 ``"small"`` 역할)다. 32B
``get_genealogist_llm_client()``는 계보 분석용이고, VRAM은 한 번에 한 모델만
허용하므로 여기서 잡으면 충돌한다.

실패는 ``None``으로 돌린다 — ``generate_digest``와 같은 계약이다. 부르는
쪽이 폴백(``derive_title``)을 쥐고 있고, 명명 실패가 저장이나 배정을 막으면
안 된다.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain._language import language_directive
from argos.brain.llm_client import OllamaClient, get_llm_client
from argos.brain.titles import MAX_TITLE_CHARS
from argos.config import settings
from argos.models.tech_event import TechEvent

logger = logging.getLogger(__name__)

MAX_EVIDENCE_DOCS = 8
"""프롬프트에 실을 근거 문서 상한. 사건이 커져도 컨텍스트가 터지지 않게 한다."""

MAX_EVIDENCE_CHARS = 400
"""근거 한 건이 차지할 수 있는 글자 수."""


@dataclass(frozen=True)
class EvidenceDoc:
    """명명 근거 한 건 — 그 사건에 매달린 문서의 제목과 요약."""

    title: str | None
    summary: str | None


@dataclass(frozen=True)
class EventNaming:
    """LLM이 지어낸 사건 이름과 요약."""

    title: str
    summary: str | None


_PROMPT = """You are a news editor naming a real-world event. Below are one or more articles that all report the SAME event.

Write:
1. A TITLE that states WHAT HAPPENED — not the headline of any single article, not clickbait, not a question.
2. A SUMMARY of 1–3 sentences describing the event using only facts present below.

Rules:
- Output EXACTLY two lines, in this format, and nothing else:
TITLE: <the title>
SUMMARY: <the summary>
- No markdown, no code fences, no bullet lists, no preamble like "Here is".
- Use ONLY facts present in the articles. Do NOT invent details, numbers, or claims.

Articles:
{evidence}{language_reminder}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^\s*TITLE\s*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\s*SUMMARY\s*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_MARKUP_RE = re.compile(r"[*_`#]")


def _clean(raw: str) -> str:
    """``<think>`` 블록과 코드 펜스 줄을 걷어낸다 (digest 노드와 같은 성격)."""
    return _FENCE_RE.sub("", _THINK_RE.sub("", raw)).strip()


def _scrub(value: str) -> str:
    """남은 마크다운 강조 문자와 잔여 백틱을 지우고 공백을 접는다."""
    return re.sub(r"\s+", " ", _MARKUP_RE.sub("", value)).strip()


def _format_evidence(docs: Sequence[EvidenceDoc]) -> str:
    lines: list[str] = []
    for index, doc in enumerate(docs[:MAX_EVIDENCE_DOCS], start=1):
        title = (doc.title or "").strip()[:MAX_EVIDENCE_CHARS]
        summary = (doc.summary or "").strip()[:MAX_EVIDENCE_CHARS]
        lines.append(f"[{index}] {title}\n{summary}".rstrip())
    return "\n\n".join(lines)


async def generate_event_naming(
    docs: Sequence[EvidenceDoc],
    *,
    client: OllamaClient | None = None,
    keep_alive: str | int = "5m",
) -> EventNaming | None:
    """근거 문서들로 사건 제목·요약을 짓는다. 실패하면 ``None``.

    ``None``이 되는 경우: 근거가 하나도 없음 / LLM 호출 실패 / 정제 후 제목이
    비어 있음. 부르는 쪽은 ``None``을 받으면 ``derive_title`` 폴백을 쓰고
    ``naming_stale``을 세워 둔 채 넘어간다.

    ``client``를 넘기면 재사용한다 — 백필이 배치 전체에서 모델을 한 번만
    적재하기 위해서다. ``keep_alive``도 그 목적으로 열어 둔다.
    """
    evidence_docs = [doc for doc in docs if (doc.title or doc.summary)]
    if not evidence_docs:
        return None

    if client is None:
        client = get_llm_client()
    language = settings.user.slack.summary_language or "English"
    prompt = _PROMPT.format(
        evidence=_format_evidence(evidence_docs),
        language_reminder=language_directive(language),
    )

    try:
        raw = await client.query("small", prompt, keep_alive=keep_alive, think=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_event_naming failed: %r", exc)
        return None

    cleaned = _clean(raw or "")
    title_match = _TITLE_RE.search(cleaned)
    if title_match is None:
        logger.warning("generate_event_naming: no TITLE line in the model output")
        return None
    title = _scrub(title_match.group(1))[:MAX_TITLE_CHARS]
    if not title:
        return None

    summary_match = _SUMMARY_RE.search(cleaned)
    summary = _scrub(summary_match.group(1)) if summary_match else None
    return EventNaming(title=title, summary=summary or None)


async def apply_event_naming(
    session: AsyncSession,
    event_id: uuid.UUID,
    docs: Sequence[EvidenceDoc],
    *,
    client: OllamaClient | None = None,
    keep_alive: str | int = "5m",
) -> bool:
    """사건에 이름·요약을 지어 붙이고 ``naming_stale``을 내린다.

    생성이 실패하면 **아무것도 쓰지 않고** ``False``를 돌린다 — 사건은 무명 +
    ``naming_stale=True``인 채로 남아 뒤의 ``--rename-stale`` 패스가 줍는다.
    이 함수는 예외를 삼키지 않는다(생성 실패는 ``generate_event_naming``이
    이미 ``None``으로 접어 준다). 세이브포인트는 부르는 쪽이 건다.
    """
    naming = await generate_event_naming(docs, client=client, keep_alive=keep_alive)
    if naming is None:
        return False

    await session.execute(
        update(TechEvent)
        .where(TechEvent.id == event_id)
        .values(title=naming.title, summary=naming.summary, naming_stale=False)
    )
    return True
