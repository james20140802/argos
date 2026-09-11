"""툼스톤 체인 해석 — 흡수된 사건 id를 최종 사건 id로 바꾼다.

병합이 삭제가 아니라 툼스톤이므로(``merged_into_id``만 채운다), 옛 id로 들어온
조회는 체인을 따라가 살아 있는 사건에 도달해야 한다. 이 헬퍼는 앞으로 **모든**
사건 조회에 끼므로, 이상한 데이터(순환, 지나치게 긴 체인)를 만나도 예외를 던져
피드를 죽이지 않고 마지막으로 도달한 id를 돌려주며 경고만 남긴다 (A6).

코어(``resolve_event_chain``)는 세션을 모르고 "id → merged_into_id" 조회를
주입받는다. 덕분에 Postgres 없이 dict 기반 가짜 조회로 전 동작이 단위 테스트된다.

주의: 여기서 해석하는 건 ``TechEvent`` id 하나뿐이다. ``EventDocument``/
``EventEntity`` 링크를 흡수된 사건에서 생존 사건으로 옮기는 일은 하지 않는다
— 그건 병합 작성자(merge-writer, 후속 이슈)의 몫이다. 자세한 내용은
``services/events.py``, ``services/entities.py``의 모듈 docstring 참고.
"""
from __future__ import annotations

import logging
import uuid
from typing import Awaitable, Callable, Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_event import TechEvent

logger = logging.getLogger(__name__)

MAX_MERGE_HOPS = 8
"""툼스톤 체인을 따라갈 최대 단계. 이걸 넘으면 멈추고 경고한다."""

MergedIntoFetcher = Callable[[uuid.UUID], Awaitable[uuid.UUID | None]]

MergedIntoBatchFetcher = Callable[
    [frozenset], Awaitable[Mapping[uuid.UUID, "uuid.UUID | None"]]
]


async def resolve_event_chain(
    start_id: uuid.UUID,
    fetch_merged_into: MergedIntoFetcher,
    max_hops: int = MAX_MERGE_HOPS,
) -> uuid.UUID:
    """``start_id``에서 툼스톤 체인을 따라가 최종 사건 id를 반환한다.

    Args:
        start_id: 조회에 들어온 사건 id (흡수된 쪽일 수 있다).
        fetch_merged_into: 사건 id를 받아 ``merged_into_id``(없으면 None)를
            돌려주는 async 함수. 세션은 이 함수 안에 갇힌다.
        max_hops: 따라갈 최대 단계 수.

    Returns:
        살아 있는 사건 id. 순환이거나 ``max_hops``를 넘으면 **마지막으로 도달한
        id**를 돌려준다 — 예외를 던지지 않는다.
    """
    current = start_id
    visited = {start_id}

    for _ in range(max_hops):
        next_id = await fetch_merged_into(current)
        if next_id is None:
            return current
        if next_id in visited:
            logger.warning(
                "Tombstone chain from %s cycles back to %s; stopping at %s.",
                start_id,
                next_id,
                current,
            )
            return current
        visited.add(next_id)
        current = next_id

    logger.warning(
        "Tombstone chain from %s exceeded %d hops; stopping at %s.",
        start_id,
        max_hops,
        current,
    )
    return current


async def resolve_event(session: AsyncSession, event_id: uuid.UUID) -> uuid.UUID:
    """``resolve_event_chain``의 얇은 DB 래퍼. 자체 로직은 없다."""

    async def _fetch(current_id: uuid.UUID) -> uuid.UUID | None:
        return await session.scalar(
            select(TechEvent.merged_into_id).where(TechEvent.id == current_id)
        )

    return await resolve_event_chain(event_id, _fetch)


async def resolve_event_chains(
    start_ids: Iterable[uuid.UUID],
    fetch_merged_into_batch: MergedIntoBatchFetcher,
    max_hops: int = MAX_MERGE_HOPS,
) -> dict[uuid.UUID, uuid.UUID]:
    """여러 사건 id를 한꺼번에 해석한다 — `resolve_event_chain`의 배치 판.

    답은 하나씩 부른 것과 **같아야 한다**(순환·한도 초과에서 마지막 도달 id를
    돌려주는 동작까지). 달라지는 건 왕복 횟수뿐이다: 사건 수가 아니라 **체인
    깊이**를 따른다. 기간 전체 재군집처럼 사건이 수백 개인 자리에서 사건당 한
    번씩 물으면 그래프 계산 전에 직렬 왕복만 수백 번이라, 나머지 입력을 다
    배치로 읽어 온 보람이 없어진다.

    깊이를 먼저 다 긁어 온 뒤 코어를 dict 조회로 돌리는 구조라, 한 번 본 id는
    다시 묻지 않는다(순환도 여기서 저절로 멈춘다). 미리 긁는 라운드 수가
    `max_hops`인 것은 하나씩 부르는 판이 정확히 그만큼 조회하기 때문이다 —
    한 라운드라도 모자라면 지나치게 긴 체인에서 답이 갈린다.

    Args:
        start_ids: 해석할 사건 id들 (흡수된 쪽일 수 있다).
        fetch_merged_into_batch: id 집합을 받아 `{id: merged_into_id | None}`를
            돌려주는 async 함수. 세션은 이 함수 안에 갇힌다.
        max_hops: 따라갈 최대 단계 수.

    Returns:
        `{물어본 id: 살아 있는 사건 id}`.
    """
    pending = set(start_ids)
    if not pending:
        return {}

    merged_into: dict[uuid.UUID, uuid.UUID | None] = {}
    frontier = set(pending)
    for _ in range(max_hops):
        unknown = frontier - merged_into.keys()
        if not unknown:
            break
        found = await fetch_merged_into_batch(frozenset(unknown))
        for event_id in unknown:
            merged_into[event_id] = found.get(event_id)
        frontier = {
            merged_into[event_id]
            for event_id in unknown
            if merged_into[event_id] is not None
        }

    async def _fetch(current_id: uuid.UUID) -> uuid.UUID | None:
        return merged_into.get(current_id)

    return {
        event_id: await resolve_event_chain(event_id, _fetch, max_hops)
        for event_id in sorted(pending)
    }


async def resolve_events(
    session: AsyncSession, event_ids: Iterable[uuid.UUID]
) -> dict[uuid.UUID, uuid.UUID]:
    """`resolve_event_chains`의 얇은 DB 래퍼. 자체 로직은 없다."""

    async def _fetch_batch(ids: frozenset) -> Mapping[uuid.UUID, uuid.UUID | None]:
        rows = await session.execute(
            select(TechEvent.id, TechEvent.merged_into_id).where(
                TechEvent.id.in_(sorted(ids))
            )
        )
        return {row.id: row.merged_into_id for row in rows}

    return await resolve_event_chains(event_ids, _fetch_batch)
