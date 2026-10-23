"""별칭 층의 DB 읽기·쓰기 — ARG-312.

판정은 ``entity_aliases``(순수 함수)가 맡고, 여기서는 (1) 저장된 이름·문서 빈도·
본문을 읽어 입력을 만들고, (2) 계산된 계획을 ``entity_aliases`` 테이블에 반영한다.

``entities``·``document_entities``는 **읽기만** 한다. 이름 저장 경로
(``entity_store``)는 건드리지 않는다 — 사건 묶기가 읽는 입력이 그대로여야 한다.
ARG-313의 하루 배치가 같은 ``plan_aliases`` / ``apply_aliases``를 불러 쓸 수 있다.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_aliases import (
    AliasInputs,
    AliasPlan,
    EntityRow,
    ExistingAlias,
    compute_alias_plan,
)
from argos.brain.entity_store import name_document_frequency
from argos.models.entity import Entity
from argos.models.entity_alias import AliasRule, EntityAlias
from argos.models.tech_item import TechItem

_INSERT_CHUNK = 1000


class AliasTableMissing(RuntimeError):
    """``entity_aliases``가 아직 없다 — ``alembic upgrade head``가 먼저다."""


async def alias_table_exists(session: AsyncSession) -> bool:
    return (
        await session.scalar(text("SELECT to_regclass('public.entity_aliases') IS NOT NULL"))
    ) is True


async def load_inputs(session: AsyncSession) -> AliasInputs:
    """읽기 전용. 테이블이 아직 없으면 기존 별칭은 비어 있는 것으로 본다."""
    rows = (await session.execute(select(Entity.id, Entity.normalized_key, Entity.name))).all()
    entities = tuple(EntityRow(id=r[0], key=r[1], name=r[2]) for r in rows)

    frequency, _ = await name_document_frequency(session)

    text_rows = (await session.execute(select(TechItem.title, TechItem.raw_content))).all()
    texts = tuple(f"{title or ''}\n{raw or ''}" for title, raw in text_rows)

    existing: tuple[ExistingAlias, ...] = ()
    if await alias_table_exists(session):
        found = (
            await session.execute(
                select(
                    EntityAlias.alias_entity_id,
                    EntityAlias.canonical_entity_id,
                    EntityAlias.rule,
                )
            )
        ).all()
        existing = tuple(ExistingAlias(r[0], r[1], AliasRule(r[2])) for r in found)

    return AliasInputs(
        entities=entities,
        doc_counts=frequency,
        texts=texts,
        existing=existing,
    )


async def plan_aliases(session: AsyncSession) -> AliasPlan:
    """별칭 계획을 만든다. SELECT만 실행한다."""
    return compute_alias_plan(await load_inputs(session))


async def apply_aliases(session: AsyncSession, plan: AliasPlan) -> int:
    """계획대로 ``Manual``이 아닌 별칭 행을 교체한다. 커밋은 호출자 몫이다.

    지울 행을 먼저 지우고 넣는다 — 대표 이름이 바뀐 별칭은 같은 트랜잭션 안에서
    (지움, 넣음) 한 쌍이 되는데, ``alias_entity_id`` UNIQUE 때문에 순서가 거꾸로면 걸린다.
    """
    if not plan.changed:
        return 0
    if not await alias_table_exists(session):
        raise AliasTableMissing(
            "entity_aliases 테이블이 없습니다. `uv run alembic upgrade head`를 먼저 실행하세요."
        )

    removed_ids: list[uuid.UUID] = [link.alias_id for link in plan.removed]
    if removed_ids:
        await session.execute(
            delete(EntityAlias).where(
                EntityAlias.alias_entity_id.in_(removed_ids),
                EntityAlias.rule != AliasRule.MANUAL,
            )
        )
    values = [
        {
            "id": uuid.uuid4(),
            "alias_entity_id": link.alias_id,
            "canonical_entity_id": link.canonical_id,
            "rule": link.rule,
            "evidence": dict(link.evidence),
        }
        for link in plan.added
    ]
    for start in range(0, len(values), _INSERT_CHUNK):
        await session.execute(pg_insert(EntityAlias).values(values[start : start + _INSERT_CHUNK]))
    return len(plan.added) + len(plan.removed)
