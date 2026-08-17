"""이름 사전(가제티어) 조회 — 사건→이름, 이름→사건 양방향.

정규화 함수를 순수 함수로 떼어 둔 이유는 두 가지다. (1) DB 없이 단위 테스트가
돌아야 하고, (2) 형제 이슈 ARG-225가 정규화 산식을 바꿀 때 갈아끼울 자리가
한 곳이어야 하기 때문이다. 산식이 바뀌면 ``normalized_key`` 재계산 백필이
한 번 필요하다.
"""
from __future__ import annotations

import uuid

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.entity import Entity, EventEntity
from argos.models.tech_event import TechEvent


def normalize_entity_name(raw: str) -> str:
    """이름을 사전 키로 바꾼다 — 소문자화 + 공백 정리까지만 (A3).

    하이픈/구두점 변형 통합(``Sonnet-5`` ↔ ``sonnet 5``)은 여기서 하지 않는다.
    그건 형제 이슈 ARG-225의 범위다.
    """
    return " ".join(raw.split()).lower()


def build_event_entities_query(event_id: uuid.UUID) -> Select:
    """사건 id → 그 사건에 붙은 이름 목록을 뽑는 SELECT."""
    return (
        select(Entity)
        .join(EventEntity, EventEntity.entity_id == Entity.id)
        .where(EventEntity.event_id == event_id)
        .order_by(Entity.name)
    )


def build_events_for_entity_query(normalized_key: str) -> Select:
    """정규화 키 → 그 이름이 붙은 사건 목록을 뽑는 SELECT (역조회)."""
    return (
        select(TechEvent)
        .join(EventEntity, EventEntity.event_id == TechEvent.id)
        .join(Entity, Entity.id == EventEntity.entity_id)
        .where(Entity.normalized_key == normalized_key)
        .order_by(TechEvent.occurred_at.desc())
    )


async def list_event_entities(
    session: AsyncSession,
    event_id: uuid.UUID,
) -> list[Entity]:
    """사건에 붙은 이름 목록. 없으면 빈 리스트."""
    result = await session.execute(build_event_entities_query(event_id))
    return list(result.scalars().all())


async def list_events_for_entity(
    session: AsyncSession,
    name: str,
) -> list[TechEvent]:
    """이름으로 사건을 역조회한다.

    **원문 이름**을 받아 내부에서 정규화한다 — 호출자가 정규화를 잊어서 조용히
    빈 결과를 받는 사고를 구조적으로 없애기 위함이다.
    """
    query = build_events_for_entity_query(normalize_entity_name(name))
    result = await session.execute(query)
    return list(result.scalars().all())
