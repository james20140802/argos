"""문서에 이름을 매다는 쓰기 경로 — ARG-263.

추출 산식은 ``entity_extraction`` 소관이고 여기서 재구현하지 않는다. 이
모듈이 맡는 건 (1) 추출 결과를 ``entities`` 가제티어에 정규화 키로 upsert,
(2) ``document_entities``에 링크, (3) 이웃 문서들의 이름을 한 번에 읽어오기
세 가지다.

정규화 키는 ``ExtractedName.canonical``을 그대로 쓴다. 추출기가 이미 소유격
접기·대소문자 접기를 마친 비교용 정규형을 내주므로, 여기서 두 번째 정규화
산식을 만들면 같은 이름이 두 경로에서 다르게 갈린다.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_extraction import ExtractedName
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity

logger = logging.getLogger(__name__)


def storage_key(name: ExtractedName) -> str:
    return name.canonical


def canonical_names(names: Iterable[ExtractedName]) -> list[str]:
    """중복을 접고 정렬한 정규형 목록. 같은 입력에 항상 같은 출력."""
    return sorted({storage_key(name) for name in names})


async def attach_names(
    session: AsyncSession,
    tech_item_id: uuid.UUID,
    names: Sequence[ExtractedName],
) -> None:
    """이름을 가제티어에 넣고 문서에 매단다. 여러 번 불러도 결과가 같다.

    ``ON CONFLICT DO NOTHING``을 쓰는 이유는 재크롤·동시 실행 때문이다.
    SELECT로 존재를 확인하고 INSERT하면 두 실행 사이에 끼어들 자리가 생겨
    유니크 위반이 터지고, 그 예외가 배치 전체의 savepoint를 말아 올린다.
    """
    if not names:
        return

    # 표시용 원문은 같은 키의 첫 등장분을 쓴다 — 정렬된 키 순서로 고정해
    # 같은 배치에서 항상 같은 원문이 저장되게 한다.
    surface_by_key: dict[str, str] = {}
    for name in names:
        surface_by_key.setdefault(storage_key(name), name.surface)
    keys = sorted(surface_by_key)

    await session.execute(
        pg_insert(Entity)
        .values([
            {"id": uuid.uuid4(), "name": surface_by_key[key], "normalized_key": key}
            for key in keys
        ])
        .on_conflict_do_nothing(index_elements=["normalized_key"]),
    )

    rows = await session.execute(
        select(Entity.id, Entity.normalized_key).where(Entity.normalized_key.in_(keys))
    )
    entity_ids = [entity_id for entity_id, _ in rows.all()]
    if not entity_ids:
        return

    await session.execute(
        pg_insert(DocumentEntity)
        .values([
            {"id": uuid.uuid4(), "tech_item_id": tech_item_id, "entity_id": entity_id}
            for entity_id in sorted(entity_ids, key=str)
        ])
        .on_conflict_do_nothing(constraint="uq_document_entities_item_entity"),
    )


async def names_for_documents(
    session: AsyncSession,
    tech_item_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, frozenset[str]]:
    """이웃 문서들의 이름을 한 번의 쿼리로. 문서당 한 번씩 도는 N+1을 피한다."""
    if not tech_item_ids:
        return {}

    rows = await session.execute(
        select(DocumentEntity.tech_item_id, Entity.normalized_key)
        .join(Entity, Entity.id == DocumentEntity.entity_id)
        .where(DocumentEntity.tech_item_id.in_(list(tech_item_ids)))
    )
    collected: dict[uuid.UUID, set[str]] = {}
    for tech_item_id, key in rows.all():
        collected.setdefault(tech_item_id, set()).add(key)
    return {key: frozenset(value) for key, value in collected.items()}
