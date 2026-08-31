"""배정 코어 — 판정과 링크 쓰기. ``BrainState``를 모른다 (ARG-270).

온라인 배정(``assign_event_node``/``save_node``)과 소급 배정
(``event_backfill``)이 **문자 그대로 같은 코드**를 부르게 하려고 뽑아냈다.
경로마다 자기 산식을 갖게 되면 백필이 만든 사건 경계와 이후 들어오는 기사의
경계가 어긋나고, 부모 AC("배정 기준이 온라인 배정과 동일")가 깨진다.

세 조각으로 나뉜다:

- ``decide_event`` — **순수 동기 함수.** 이미 손에 든 후보로 점수를 합산해
  사건을 고른다. DB도 세션도 모른다. 그래서 미리보기가 DB를 한 번도 건드리지
  않고 실행과 같은 판정을 낼 수 있다.
- ``db_candidate_source`` — 후보를 DB에서 읽는 기본 소스. 미리보기는 이걸
  감싸 인메모리 오버레이를 얹는다.
- ``link_document_to_event`` — 판정 결과를 링크로 쓴다. 여기서만
  ``naming_stale``을 세운다.

**``naming_stale``을 세우는 자리가 여기인 이유:** 사건의 경계가 바뀌면
(문서가 새로 붙으면) 기존 이름이 더 이상 그 사건 전체를 대표하지 못할 수
있다. 경계를 바꾸는 유일한 자리가 이 쓰기 함수이므로, 표시도 여기서 한다 —
온라인·소급 어느 쪽으로 들어와도 같은 표시가 남는다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.event_candidates import CandidateNeighbor, fetch_candidates
from argos.brain.event_scoring import (
    DocumentFeatures,
    EdgeWeights,
    NeighborEdge,
    choose_event,
    edge_weight,
)
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent

if TYPE_CHECKING:
    from argos.config import EventDetectionConfig


@dataclass(frozen=True)
class LinkResult:
    """링크 쓰기의 결과 — 붙은 사건과 그 사건이 이번에 새로 생겼는지."""

    event_id: uuid.UUID
    created: bool


def decide_event(
    features: DocumentFeatures,
    candidates: Sequence[CandidateNeighbor],
    *,
    config: "EventDetectionConfig",
) -> uuid.UUID | None:
    """*features*가 붙을 기존 사건을 고른다. 없으면 ``None``.

    ``None``은 "임계값을 넘는 기존 사건이 없다" = **새 사건이 필요하다**는
    뜻이다. "판정에 실패했다"는 뜻이 아니다 — 그 구분은 부르는 쪽이
    ``event_assigned``로 따로 표현한다 (``nodes/assign_event.py`` docstring).
    """
    weights = EdgeWeights.from_config(config)
    edges = [
        NeighborEdge(
            event_ids=candidate.event_ids,
            weight=edge_weight(
                features,
                candidate.features,
                weights=weights,
                window_days=config.window_days,
            ),
        )
        for candidate in candidates
        if candidate.event_ids
    ]
    return choose_event(edges, join_threshold=config.join_threshold)


async def db_candidate_source(
    session: AsyncSession,
    *,
    embedding: Sequence[float],
    at: datetime,
    exclude_id: uuid.UUID | None = None,
) -> list[CandidateNeighbor]:
    """DB에서 후보 이웃을 읽는다. 세이브포인트 안에서 부른다.

    DB 예외(예: 임베딩 차원 불일치로 인한 ``InFailedSQLTransactionError``)를
    부르는 쪽이 삼킬 수 있으려면, 그 예외가 중단시킨 트랜잭션을 되돌릴
    세이브포인트가 필요하다 (``nodes/assign_event.py`` 모듈 docstring).
    """
    async with session.begin_nested():
        return await fetch_candidates(
            session, embedding=embedding, at=at, exclude_id=exclude_id
        )


async def link_document_to_event(
    session: AsyncSession,
    *,
    tech_item_id: uuid.UUID,
    event_id: uuid.UUID | None,
    occurred_at: datetime,
) -> LinkResult:
    """문서를 사건에 매단다. ``event_id``가 ``None``이면 새 사건을 만든다.

    새 사건은 ``naming_stale=True``로 태어난다 — 아직 이름이 없으므로 명명
    경로가 주워야 한다. 기존 사건에 붙는 경우도 그 사건을
    ``naming_stale=True``로 세운다: 근거가 늘어 경계가 바뀌었으니 이름을 다시
    지어야 한다.

    링크 INSERT는 ``on_conflict_do_nothing``이라 같은 (사건, 문서) 쌍을 두 번
    써도 안전하다 — 백필 재실행의 멱등성이 여기에 걸려 있다.

    이 함수는 예외를 삼키지 않는다. 삼킬지 말지는 부르는 쪽의 정책이고,
    삼키는 쪽은 반드시 세이브포인트 안에서 불러야 한다.
    """
    created = False
    if event_id is None:
        event = TechEvent(
            id=uuid.uuid4(),
            occurred_at=occurred_at,
            naming_stale=True,
        )
        session.add(event)
        event_id = event.id
        created = True

    await session.execute(
        pg_insert(EventDocument)
        .values(id=uuid.uuid4(), event_id=event_id, tech_item_id=tech_item_id)
        .on_conflict_do_nothing(constraint="uq_event_documents_event_item"),
    )
    if not created:
        await session.execute(
            update(TechEvent)
            .where(TechEvent.id == event_id)
            .values(naming_stale=True)
        )
    return LinkResult(event_id=event_id, created=created)
