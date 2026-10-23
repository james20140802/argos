"""역할 갈래 초안 생성 — ARG-311.

``argos taxonomy draft``가 쓴다. 코퍼스에서 자주 나온 이름(엔티티)과 그 이름이 든
문장을 모아 LLM에게 (a) 이름마다 역할 한 줄, (b) 그 역할들로 트리 설계를 시킨다.
결과는 **파일로만** 나가고 DB는 건드리지 않는다 — 확정은 사람이 보고
``argos taxonomy apply``로 한다. 이 모듈은 읽기 전용이다.

``keep_top``에 트리를 주면 그 트리의 위 갈래(이름·설명·id)를 고정하고 아래 갈래만
새로 제안받는다.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_frequency import common_names
from argos.brain.entity_store import name_document_frequency
from argos.brain.taxonomy import (
    MAX_EXAMPLES,
    Branch,
    TaxonomyError,
    Tree,
    validate,
)
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.tech_item import TechItem

logger = logging.getLogger(__name__)

MAX_SENTENCES = 3
MAX_SENTENCE_CHARS = 300
_DOCS_PER_ENTITY = 5
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


class TaxonomyDraftError(RuntimeError):
    """LLM이 검증을 통과하는 트리를 끝내 만들지 못했다."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class Candidate:
    name: str
    sentences: tuple[str, ...]


# --------------------------------------------------------------------- #
# 입력 수집
# --------------------------------------------------------------------- #
def extract_sentences(
    texts: list[str],
    name: str,
    *,
    limit: int = MAX_SENTENCES,
    max_chars: int = MAX_SENTENCE_CHARS,
) -> list[str]:
    """``name``이 이름 경계로 든 문장 최대 ``limit``개 (각 ``max_chars``자 이하)."""
    pattern = re.compile(
        rf"(?<![0-9A-Za-z]){re.escape(name)}(?![0-9A-Za-z])", re.IGNORECASE
    )
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE_SPLIT.split(text or ""):
            sentence = sentence.strip()
            if not sentence or not pattern.search(sentence):
                continue
            sentence = sentence[:max_chars]
            if sentence in seen:
                continue
            seen.add(sentence)
            found.append(sentence)
            if len(found) >= limit:
                return found
    return found


async def fetch_candidates(session: AsyncSession, limit: int) -> list[Candidate]:
    """이름 있는 문서 수 기준 상위 ``limit``개 이름 + 이름이 든 문장. 읽기 전용."""
    frequency, named_documents = await name_document_frequency(session)
    cfg = settings.user.event_detection
    common = common_names(
        document_frequency=frequency,
        named_documents=named_documents,
        max_ratio=cfg.entity_common_doc_ratio,
        min_documents=cfg.entity_common_min_documents,
    )
    ranked = sorted(
        ((k, c) for k, c in frequency.items() if k not in common),
        key=lambda kc: (-kc[1], kc[0]),
    )[:limit]
    if not ranked:
        return []
    keys = [k for k, _c in ranked]
    rows = (
        await session.execute(
            select(Entity.id, Entity.name, Entity.normalized_key).where(
                Entity.normalized_key.in_(keys)
            )
        )
    ).all()
    by_key = {key: (eid, name) for eid, name, key in rows}

    out: list[Candidate] = []
    for key in keys:
        if key not in by_key:
            continue
        entity_id, name = by_key[key]
        docs = (
            await session.execute(
                select(TechItem.raw_content, TechItem.summary)
                .join(DocumentEntity, DocumentEntity.tech_item_id == TechItem.id)
                .where(DocumentEntity.entity_id == entity_id)
                .order_by(TechItem.created_at.desc())
                .limit(_DOCS_PER_ENTITY)
            )
        ).all()
        texts = [t for raw, summary in docs for t in (summary, raw) if t]
        out.append(Candidate(name=name, sentences=tuple(extract_sentences(texts, name))))
    return out


# --------------------------------------------------------------------- #
# LLM 호출
# --------------------------------------------------------------------- #
_ROLE_PROMPT = """아래 기술 이름이 문장들에서 어떤 "역할"을 하는지 한 줄(한국어, 40자 이내)로 요약하라.
제품명이 아니라 역할로 답하라 (예: "로컬에서 모델을 실행하는 러너", "코드 변경 이력을 관리하는 도구").
JSON만 출력: {{"role": "..."}}

기술 이름: {name}
문장:
{sentences}"""

_TREE_PROMPT = """아래는 기술 {count}개와 각각의 역할 요약이다. 이 기술들을 "무슨 역할인가" 기준으로 나누는 2단 분류 트리를 설계하라.

규칙:
- 위 갈래 3~6개, 위 갈래마다 아래 갈래 1~5개.
- 이름은 한국어 + 필요하면 괄호 영문. 사람이 읽는 이름이어야 하며 단어 나열("claude, went, seconds")은 안 된다.
- 설명은 한국어 한 문장 (10자 이상).
- 제품 계통이나 회사가 아니라 역할로 나눈다. "AI", "LLM", "기타" 같은 포괄적인 이름은 위 갈래로 쓰지 않는다.
- 아래 갈래마다 위 목록에서 고른 예시 기술 이름 최대 {max_examples}개를 "examples"에 넣는다 (목록에 있는 이름만).
{fixed_rule}
JSON만 출력:
{{"tree": [{{"name": "...", "description": "...", "children": [{{"name": "...", "description": "...", "examples": ["..."]}}]}}]}}

기술과 역할:
{roles}"""

_FIXED_RULE = """- 위 갈래는 아래 목록으로 **고정**이다. 이름을 바꾸거나 더하거나 빼지 말고, 각 위 갈래의 아래 갈래만 설계하라.
  고정 위 갈래: {tops}
"""


def _parse_json(raw: str) -> Any:
    text = _THINK_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("JSON 객체를 찾을 수 없음")
    return json.loads(text[start : end + 1])


def _parse_role(raw: str) -> str:
    try:
        role = _parse_json(raw).get("role", "")
    except (ValueError, AttributeError):
        return ""
    return str(role).strip()[:80]


def _tree_from_llm(
    data: Any, known: set[str], fixed: Tree | None
) -> Tree:
    if not isinstance(data, dict) or not isinstance(data.get("tree"), list):
        raise TaxonomyError(["'tree' 목록이 없습니다"])

    def children_of(item: dict, old_children: dict[str, Branch]) -> tuple[Branch, ...]:
        out: list[Branch] = []
        for c in item.get("children") or []:
            if not isinstance(c, dict) or not c.get("name"):
                raise TaxonomyError(["아래 갈래 항목에 name이 없습니다"])
            name = str(c["name"]).strip()
            examples = tuple(
                e
                for e in (str(x).strip() for x in c.get("examples") or [])
                if e in known
            )[:MAX_EXAMPLES]
            old = old_children.get(name)
            out.append(
                Branch(
                    name=name,
                    description=str(c.get("description") or "").strip(),
                    id=old.id if old else None,
                    examples=examples,
                )
            )
        return tuple(out)

    if fixed is None:
        tops = []
        for item in data["tree"]:
            if not isinstance(item, dict) or not item.get("name"):
                raise TaxonomyError(["위 갈래 항목에 name이 없습니다"])
            tops.append(
                Branch(
                    name=str(item["name"]).strip(),
                    description=str(item.get("description") or "").strip(),
                    children=children_of(item, {}),
                )
            )
        return tuple(tops)

    by_name = {str(i.get("name")).strip(): i for i in data["tree"] if isinstance(i, dict)}
    missing = [t.name for t in fixed if t.name not in by_name]
    if missing:
        raise TaxonomyError([f"고정 위 갈래가 응답에 없습니다: {m!r}" for m in missing])
    return tuple(
        replace(
            top,
            children=children_of(
                by_name[top.name], {c.name: c for c in top.children}
            ),
            examples=(),
        )
        for top in fixed
    )


async def build_draft(
    session: AsyncSession,
    llm: Any,
    *,
    limit: int | None = None,
    keep_top: Tree | None = None,
) -> Tree:
    """초안 트리를 만든다. 읽기 전용; LLM 호출은 (엔티티 수 + 1)회, 끝에 모델을 내린다."""
    cfg = settings.user.taxonomy
    candidates = await fetch_candidates(session, limit or cfg.draft_limit)
    if not candidates:
        raise TaxonomyDraftError(["초안을 만들 이름이 없습니다 — 먼저 argos backfill-entities를 실행하세요"])

    try:
        roles: list[tuple[str, str]] = []
        for cand in candidates:
            prompt = _ROLE_PROMPT.format(
                name=cand.name,
                sentences="\n".join(f"- {s}" for s in cand.sentences) or "(문장 없음)",
            )
            raw = await llm.query(
                "large", prompt, keep_alive="10m", num_ctx=cfg.num_ctx,
                think=False, temperature=0,
            )
            roles.append((cand.name, _parse_role(raw)))

        known = {name for name, _ in roles}
        listing = "\n".join(f"- {n}: {r or '(역할 불명)'}" for n, r in roles)
        fixed_rule = (
            _FIXED_RULE.format(tops=", ".join(t.name for t in keep_top))
            if keep_top
            else ""
        )
        prompt = _TREE_PROMPT.format(
            count=len(roles), max_examples=MAX_EXAMPLES,
            fixed_rule=fixed_rule, roles=listing,
        )

        errors: list[str] = []
        for _attempt in range(2):
            raw = await llm.query(
                "large", prompt, keep_alive="10m", num_ctx=cfg.num_ctx,
                think=False, temperature=0,
            )
            try:
                tree = _tree_from_llm(_parse_json(raw), known, keep_top)
            except (ValueError, TaxonomyError) as exc:
                errors = getattr(exc, "errors", None) or [str(exc)]
                continue
            errors = validate(tree)
            if not errors:
                return tree
        raise TaxonomyDraftError(errors)
    finally:
        try:
            await llm.unload("large")
        except Exception as exc:  # noqa: BLE001 — 언로드 실패가 결과를 가리면 안 된다
            logger.warning("taxonomy draft: unload failed: %r", exc)
