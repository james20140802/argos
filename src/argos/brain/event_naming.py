"""사건 이름·요약 생성 — ARG-269.

**여기가 사건 층에서 LLM이 들어오는 유일한 자리다.** 묶기 판단(edge_weight /
choose_event)에는 LLM을 쓰지 않는다 — 같은 입력이 항상 같은 배정을 내야
하는 결정성 요구 때문이다. 이름과 요약은 사람이 읽는 산출물이라 그 제약이
없다.

ARG-310: 모델은 ``event_summary.model``(기본 ``gemma4:26b``)이고
``get_event_summary_llm_client()``의 ``"large"`` 슬롯으로 부른다. VRAM은 한 번에
한 모델만 허용하므로 호출은 ``exclusive=True``다 -- 적재돼 있는 다른 모델을 내린
뒤에 올린다. 근거는 제목·요약에 더해 기사 본문까지 넣고, 컨텍스트
(``event_summary.num_ctx``)를 넘기면 청크로 나눠 맵리듀스로 접는다
(``event_summary_input``). 메모리는 num_ctx로 고정되고 입력이 거기 맞춰진다.

실패는 ``None``으로 돌린다 — ``generate_digest``와 같은 계약이다. 부르는
쪽이 폴백(``derive_title``)을 쥐고 있고, 명명 실패가 저장이나 배정을 막으면
안 된다.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain._language import language_directive
from argos.brain.event_summary_input import (
    BLOCK_SEPARATOR,
    MAX_FOLD_DEPTH,
    budget_chars,
    pack_chunks,
)
from argos.brain.llm_client import OllamaClient, get_event_summary_llm_client
from argos.brain.titles import MAX_TITLE_CHARS
from argos.config import settings
from argos.models.tech_event import TechEvent

logger = logging.getLogger(__name__)

MAX_EVIDENCE_CHARS = 400
"""근거 한 건의 제목·요약 각각이 차지할 수 있는 글자 수. 본문은 ``event_summary.doc_max_chars``."""


@dataclass(frozen=True)
class EvidenceDoc:
    """명명 근거 한 건 — 그 사건에 매달린 문서의 제목과 요약."""

    title: str | None
    summary: str | None
    body: str | None = None
    """기사 본문(ARG-310). 없으면 제목·요약만으로 짓는다."""


@dataclass(frozen=True)
class EventNaming:
    """LLM이 지어낸 사건 이름과 요약.

    둘 다 채워졌을 때만 만들어진다 — 반쪽짜리 산출물은
    ``generate_event_naming``이 ``None``으로 접는다.
    """

    title: str
    summary: str


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

_MEMO_PROMPT = """You are preparing fact notes for a news editor. Below are articles that all report the SAME event. Write compact notes of the concrete facts they contain: who did what, when, numbers, product and company names.

Rules:
- Plain text, at most 12 short lines, no markdown, no preamble.
- Keep proper nouns, product names and code identifiers exactly as written in the articles (do not translate or transliterate them).
- Use ONLY facts present below. Do NOT invent details.

Articles:
{evidence}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^\s*TITLE\s*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\s*SUMMARY\s*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
"""줄 맨 앞의 헤딩 표시만 지운다. 문자 하나로 보면 ``C# 14``의 ``#``까지 먹는다."""

# 짝이 맞는 강조/코드 스팬만 벗긴다. 여는·닫는 기호가 공백이 아닌 글자에
# 붙어 있어야 해서, 짝 없이 떠 있는 기호(``2 * 3``, ``C#``)는 그대로 남는다.
# ``_``는 아예 건드리지 않는다 — ``snake_case``·``__init__``을 깨뜨리는 값이
# 잃는 것(밑줄 이탤릭 정리)보다 크다.
_BOLD_RE = re.compile(r"\*\*(\S(?:.*?\S)?)\*\*")
_ITALIC_RE = re.compile(r"\*(\S(?:[^*]*?\S)?)\*")
_CODE_RE = re.compile(r"`(\S(?:[^`]*?\S)?)`")


def _clean(raw: str) -> str:
    """``<think>`` 블록과 코드 펜스 줄을 걷어낸다 (digest 노드와 같은 성격)."""
    return _FENCE_RE.sub("", _THINK_RE.sub("", raw)).strip()


def _scrub(value: str) -> str:
    """남은 마크다운 껍데기를 벗기고 공백을 접는다.

    프롬프트가 이미 "no markdown"을 요구하므로 여기는 잔여물 정리다. 기호를
    문자 단위로 지우면 안 된다 — ``C# 14 출시``가 ``C 14 출시``가 되고
    ``snake_case``가 ``snakecase``가 된다(ARG-274 리뷰). 그래서 헤딩 표시는
    줄 맨 앞에서만, 강조·코드는 **짝이 맞을 때만** 벗긴다.
    """
    value = _HEADING_RE.sub("", value)
    value = _BOLD_RE.sub(r"\1", value)
    value = _ITALIC_RE.sub(r"\1", value)
    value = _CODE_RE.sub(r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def _format_block(index: int, doc: EvidenceDoc, doc_max_chars: int) -> str:
    title = (doc.title or "").strip()[:MAX_EVIDENCE_CHARS]
    summary = (doc.summary or "").strip()[:MAX_EVIDENCE_CHARS]
    body = (doc.body or "").strip()[:doc_max_chars]
    return "\n".join(part for part in (f"[{index}] {title}", summary, body) if part)


def _format_evidence(docs: Sequence[EvidenceDoc], doc_max_chars: int, max_docs: int) -> list[str]:
    """문서마다 한 블록 (제목 + 요약 + 본문). 앞에서부터 ``max_docs``건."""
    return [
        _format_block(index, doc, doc_max_chars)
        for index, doc in enumerate(docs[:max_docs], start=1)
    ]


def _parse_naming(raw: str) -> EventNaming | None:
    cleaned = _clean(raw or "")
    title_match = _TITLE_RE.search(cleaned)
    if title_match is None:
        logger.warning("generate_event_naming: no TITLE line in the model output")
        return None
    title = _scrub(title_match.group(1))[:MAX_TITLE_CHARS]
    if not title:
        return None

    summary_match = _SUMMARY_RE.search(cleaned)
    summary = _scrub(summary_match.group(1)) if summary_match else ""
    if not summary:
        # 제목만 온 출력도 실패다. 통과시키면 apply_event_naming이
        # summary=NULL을 쓰면서 naming_stale까지 내려, 이미 붙어 있던 요약을
        # 지우고 다음 --rename-stale 패스가 주울 기회까지 없앤다. 제목이 빈
        # 경우와 같은 처리로 접어야 재생성 여지가 남는다.
        logger.warning("generate_event_naming: no SUMMARY line in the model output")
        return None
    return EventNaming(title=title, summary=summary)


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
) -> EventNaming | None:
    """근거 문서들로 사건 제목·요약을 짓는다. 실패하면 ``None``.

    ``None``이 되는 경우: 근거가 하나도 없음 / LLM 호출 실패 / 정제 후 제목이
    비어 있음 / 정제 후 요약이 비어 있음 / 맵리듀스가 ``MAX_FOLD_DEPTH``단계
    안에 안 접힘. 부르는 쪽은 ``None``을 받으면 ``derive_title`` 폴백을 쓰고
    ``naming_stale``을 세워 둔 채 넘어간다.

    근거가 컨텍스트 예산 안이면 한 번 부르고, 넘치면 청크마다 사실 메모를 뽑은
    뒤 메모들로 최종 TITLE/SUMMARY를 짓는다. 모든 호출은 같은 모델·같은
    ``num_ctx``·``exclusive=True``라 재적재가 없다.

    ``client``를 넘기면 재사용한다 -- 백필이 배치 전체에서 모델을 한 번만
    적재하기 위해서다. ``keep_alive``도 그 목적으로 열어 둔다. ``label``은
    로그에 찍을 사건 식별자다.
    """
    evidence_docs = [doc for doc in docs if (doc.title or doc.summary or doc.body)]
    if not evidence_docs:
        return None

    cfg = settings.user.event_summary
    if client is None:
        client = get_event_summary_llm_client()
    language = settings.user.slack.summary_language or "English"
    reminder = language_directive(language)

    template_chars = max(len(_PROMPT), len(_MEMO_PROMPT)) + len(reminder)
    budget = budget_chars(
        num_ctx=cfg.num_ctx,
        output_reserve_tok=cfg.output_reserve_tok,
        chars_per_tok=cfg.chars_per_tok,
        template_chars=template_chars,
    )
    log = _CallLog(cfg.num_ctx)

    async def _ask(prompt: str) -> str:
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
        )
        log.add(stats)
        return raw

    blocks = _format_evidence(evidence_docs, cfg.doc_max_chars, cfg.max_docs)
    chunk_count = 1
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
                raw = await _ask(_MEMO_PROMPT.format(evidence=BLOCK_SEPARATOR.join(chunk)))
                memo = _clean(raw or "")[:memo_cap]
                if not memo:
                    logger.warning("generate_event_naming: empty fact memo")
                    return None
                memos.append(f"[notes {len(memos) + 1}]\n{memo}")
            items = memos
            depth += 1
        evidence = BLOCK_SEPARATOR.join(chunks[0]) if chunks else ""
        raw = await _ask(_PROMPT.format(evidence=evidence, language_reminder=reminder))
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_event_naming failed: %r", exc)
        return None

    logger.info(
        "event naming %s: docs=%d chunks=%d calls=%d prompt_tokens=%d seconds=%.1f",
        label or "-",
        len(blocks),
        chunk_count,
        log.calls,
        log.prompt_tokens,
        time.monotonic() - log.started,
    )
    return _parse_naming(raw)


async def apply_event_naming(
    session: AsyncSession,
    event_id: uuid.UUID,
    docs: Sequence[EvidenceDoc],
    *,
    client: OllamaClient | None = None,
    keep_alive: str | int = "5m",
    expected_updated_at: datetime | None = None,
) -> bool:
    """사건에 이름·요약을 지어 붙이고 ``naming_stale``을 내린다.

    생성이 실패하면 **아무것도 쓰지 않고** ``False``를 돌린다 — 사건은 무명 +
    ``naming_stale=True``인 채로 남아 뒤의 ``--rename-stale`` 패스가 줍는다.
    이 함수는 예외를 삼키지 않는다(생성 실패는 ``generate_event_naming``이
    이미 ``None``으로 접어 준다). 세이브포인트는 부르는 쪽이 건다.

    ARG-274: ``expected_updated_at``을 주면 그 값이 아직 그대로일 때만 쓴다.
    근거를 스냅샷으로 뜬 뒤 LLM이 도는 동안(초 단위) 온라인 파이프라인이 같은
    사건에 문서를 하나 더 매달 수 있는데, 그러면
    ``link_document_to_event``가 ``naming_stale``을 다시 세운다. 가드가 없으면
    옛 근거로 지은 이름을 쓰면서 그 플래그까지 내려 버려, 새로 붙은 문서가
    이름에도 안 들어가고 다음 ``--rename-stale``에도 안 걸린다. 가드가 걸리면
    아무것도 쓰지 않고 ``False``를 돌린다 — 위 "실패하면 stale로 남는다"와
    같은 처리라 다음 패스가 새 근거까지 포함해 다시 짓는다.

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
        stmt.values(title=naming.title, summary=naming.summary, naming_stale=False)
    )
    if expected_updated_at is not None and result.rowcount == 0:
        logger.info(
            "apply_event_naming: %s changed while naming ran — leaving it stale",
            event_id,
        )
        return False
    return True
