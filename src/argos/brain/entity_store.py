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

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_extraction import ExtractedName
from argos.brain.entity_frequency import drop_common_names
from argos.config import settings
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


async def raw_names_for_documents(
    session: AsyncSession,
    tech_item_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, frozenset[str]]:
    """이웃 문서들의 **저장된 그대로의** 이름을 한 번의 쿼리로.

    문서당 한 번씩 도는 N+1을 피한다. 흔한 이름을 걸러내지 않는다 — 채우기
    미리보기처럼 저장 상태를 그대로 봐야 하는 곳이 쓴다.
    """
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


async def name_document_frequency(session: AsyncSession) -> tuple[dict[str, int], int]:
    """(정규형 키 -> 그 이름이 붙은 문서 수, 이름이 1개 이상인 문서 수).

    빈도는 실행 시점에 저장된 링크에서 그때그때 센다 — 스키마에 통계를 따로
    두지 않으니 이름이 갱신되면 다음 호출부터 바로 반영된다. (item, entity)가
    유니크라 그룹별 행 수가 곧 문서 수다.
    """
    frequency_rows = await session.execute(
        select(Entity.normalized_key, func.count())
        .join(DocumentEntity, DocumentEntity.entity_id == Entity.id)
        .group_by(Entity.normalized_key)
    )
    frequency = {key: count for key, count in frequency_rows.all()}
    named = await session.scalar(
        select(func.count(func.distinct(DocumentEntity.tech_item_id)))
    )
    return frequency, int(named or 0)


async def names_for_documents(
    session: AsyncSession,
    tech_item_ids: Sequence[uuid.UUID],
) -> dict[uuid.UUID, frozenset[str]]:
    """이웃 문서들의 이름. 코퍼스 전반에 흔한 이름은 걸러서 돌려준다.

    배치 안 문서빈도 컷은 크롤 날짜마다 따로 돌아서 ``ai``·``openai``가 새어
    나온다. 저장된 이름 전체의 빈도로 한 번 더 거른다(ARG-302). 이름 링크가
    하나도 없으면 빈도 통계가 비어 아무것도 지우지 않는다.
    """
    names = await raw_names_for_documents(session, tech_item_ids)
    if not names:
        return names
    frequency, named_documents = await name_document_frequency(session)
    return drop_common_names(
        names,
        document_frequency=frequency,
        named_documents=named_documents,
        max_ratio=settings.user.event_detection.entity_common_doc_ratio,
    )
