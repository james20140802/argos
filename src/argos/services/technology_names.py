"""대표 이름과 다른 이름 조회 — ARG-312.

``entity_aliases``가 이어 둔 이름 묶음에서 화면에 보여 줄 이름 하나와 "다른 이름"을
읽기 시점에 계산한다. ``entities.name``에는 키마다 **첫 등장 표기 하나**만 있어
(``openclaw`` -> "Openclaw") 그대로 쓰면 틀린 표기가 대표가 된다. 그래서 묶음에 속한
문서 본문에서 대소문자를 구분해 표기별 등장 수를 세고 가장 많은 표기를 이름으로 쓴다.
계산 결과는 저장하지 않는다.

기술 페이지(ARG-315)가 이 함수를 불러 "다른 이름"을 보여 줄 수 있다. 지금은 CLI
``argos alias-names --show``만 쓴다.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_names import canonical_name
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.entity_alias import EntityAlias
from argos.models.tech_item import TechItem

_WORD = re.compile(r"[^\W_]+")
_MAX_OTHER_NAMES = 8


@dataclass(frozen=True)
class TechnologyNames:
    """한 묶음의 보여 줄 이름과 다른 이름."""

    display: str
    aliases: tuple[str, ...]
    canonical_key: str
    member_keys: tuple[str, ...]


def surface_pattern(key: str) -> re.Pattern[str]:
    """정규형 키가 본문에서 쓰이는 표기들을 잡는 정규식(대소문자 무시).

    낱말 사이는 공백·하이픈·점·밑줄 몇 개까지 느슨하게 받는다(``Node.js`` / ``node js``).
    ``+``·``#``·앞점은 이름의 일부라 키에 있으면 표기에도 있어야 하고, 키에 없으면
    뒤에 붙어 있는 표기(``C`` vs ``C++``)는 잡지 않는다.
    """
    found = list(_WORD.finditer(key))
    if not found:
        return re.compile(r"(?!)")
    tokens = [m.group(0) for m in found]
    parts = [re.escape(tokens[0])]
    for prev, cur in zip(found, found[1:]):
        if prev.group(0).isdigit() and cur.group(0).isdigit():
            # 숫자끼리는 버전이다: 키의 구분자를 그대로 지킨다(5.5 != 55, 5.5 != 5 5).
            sep = key[prev.end() : cur.start()]
            parts.append(r"\." if "." in sep else r"[\s\-_]{1,2}")
        else:
            parts.append(r"[\s\-_.]{0,2}")
        parts.append(re.escape(cur.group(0)))
    body = "".join(parts)
    lead = r"\." if key.startswith(".") else ""
    trail_match = re.search(r"[+#]+$", key)
    trail = re.escape(trail_match.group(0)) if trail_match else r"(?![+#])"
    # 숫자로 시작/끝나는 키는 버전 조각 안에서 끊기지 않게 한다(5.5 vs 5.55 / 5.5.1).
    pre = r"(?<!\d\.)" if tokens[0][0].isdigit() and not lead else ""
    post = r"(?!\.\d)" if tokens[-1][-1].isdigit() and not trail_match else ""
    return re.compile(rf"{pre}(?<![^\W_]){lead}{body}{trail}(?![^\W_]){post}", re.IGNORECASE)


def pick_display_names(
    members: Sequence[tuple[str, str]],
    canonical_key: str,
    texts: Iterable[str],
) -> tuple[str, tuple[str, ...]]:
    """(표시 이름, 다른 이름들). ``members``는 (정규형 키, 저장된 표기) 쌍이다.

    본문에서 대소문자를 구분해 센 표기 중 가장 많은 것이 표시 이름이다(동률이면
    표기 문자열 순). 본문에서 하나도 못 찾으면 대표 이름의 저장된 표기를 쓴다.
    다른 이름은 나머지 표기(많이 쓰인 순)와 묶음의 저장된 표기들이다.
    """
    patterns = [surface_pattern(key) for key, _ in members]
    surfaces: Counter[str] = Counter()
    for text in texts:
        for pattern in patterns:
            for match in pattern.finditer(text):
                surfaces[" ".join(match.group(0).split())] += 1

    stored = {key: name for key, name in members}
    if surfaces:
        ranked = sorted(surfaces.items(), key=lambda item: (-item[1], item[0]))
        display = ranked[0][0]
        others = [surface for surface, _ in ranked[1:]]
    else:
        display = stored.get(canonical_key) or members[0][1]
        others = []
    for name in sorted(stored.values()):
        if name not in others and name != display:
            others.append(name)
    return display, tuple(others[:_MAX_OTHER_NAMES])


async def names_for(session: AsyncSession, name: str) -> TechnologyNames | None:
    """이름(어떤 표기든) -> 그 이름이 속한 묶음의 표시 이름과 다른 이름. 읽기만 한다.

    저장된 이름에 없으면 ``None``이다. 별칭 행이 없는 이름은 혼자인 묶음이다.
    """
    entity = (
        await session.execute(select(Entity).where(Entity.normalized_key == canonical_name(name)))
    ).scalar_one_or_none()
    if entity is None:
        return None

    canonical_id = await session.scalar(
        select(EntityAlias.canonical_entity_id).where(EntityAlias.alias_entity_id == entity.id)
    )
    canonical = entity
    if canonical_id is not None:
        canonical = (
            await session.execute(select(Entity).where(Entity.id == canonical_id))
        ).scalar_one()

    alias_rows = (
        await session.execute(
            select(Entity)
            .join(EntityAlias, EntityAlias.alias_entity_id == Entity.id)
            .where(EntityAlias.canonical_entity_id == canonical.id)
        )
    ).scalars().all()
    members = [canonical, *sorted(alias_rows, key=lambda e: e.normalized_key)]
    member_ids: list[uuid.UUID] = [m.id for m in members]

    text_rows = (
        await session.execute(
            select(TechItem.title, TechItem.raw_content)
            .join(DocumentEntity, DocumentEntity.tech_item_id == TechItem.id)
            .where(DocumentEntity.entity_id.in_(member_ids))
            .distinct()
        )
    ).all()
    texts = [f"{title or ''}\n{raw or ''}" for title, raw in text_rows]

    display, others = pick_display_names(
        [(m.normalized_key, m.name) for m in members], canonical.normalized_key, texts
    )
    return TechnologyNames(
        display=display,
        aliases=others,
        canonical_key=canonical.normalized_key,
        member_keys=tuple(m.normalized_key for m in members),
    )
