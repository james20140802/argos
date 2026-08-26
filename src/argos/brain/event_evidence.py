"""사건의 근거 수 집계 — ARG-267.

재배포본(같은 기사, 다른 URL)도 사건에는 매달린다(그래야 부모 성공 장면
3 "혼자인 기사도 사건 하나를 가진다"가 재배포본에도 성립한다). 다만 근거
"수"를 셀 때는 근접중복을 하나로 접는다 — 그렇지 않으면 재배포될 때마다
근거가 늘어난 것처럼 보인다.

근거 수는 ``tech_events``에 저장하지 않는다. 조회 시점에 SimHash로 계산하는
파생값이다 — 컬럼을 만들면 갱신 경로가 필요해지고, 그 갱신 경로가 어긋나는
자리가 생긴다(사건 병합, 문서 재배정 등 근거 집합이 바뀌는 모든 자리를
빠짐없이 갱신해야 하기 때문). 대신 매번 다시 계산한다 — 문서 수가 사건당
많지 않으므로 비용은 무시할 만하다.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.near_duplicate import hamming_distance
from argos.brain.simhash_storage import from_storage
from argos.config import settings
from argos.models.event_document import EventDocument
from argos.models.tech_item import TechItem


def fold_near_duplicates(hashes: Sequence[int | None], *, max_distance: int) -> int:
    """근접중복 묶음의 개수. ``None``은 각각 별개로 센다.

    정렬된 순서로 훑으며 이미 만든 묶음의 대표와 해밍 거리가 ``max_distance``
    이하면 그 묶음에 넣는 단일 연결(single-linkage) 방식이다. 완전한 군집화는
    뒤 이슈 소관이고, 재배포본 판정에는 이걸로 충분하다(설계 메모 참고).

    입력 순서에 결과가 흔들리지 않도록 해시 값 오름차순으로 먼저 정렬한다.
    """
    present = sorted(value for value in hashes if value is not None)
    missing = sum(1 for value in hashes if value is None)

    representatives: list[int] = []
    for value in present:
        if any(hamming_distance(value, rep) <= max_distance for rep in representatives):
            continue
        representatives.append(value)
    return len(representatives) + missing


async def evidence_count(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    max_distance: int | None = None,
) -> int:
    """사건의 근거 수. 저장값이 아니라 조회 시 계산하는 파생값이다."""
    if max_distance is None:
        max_distance = settings.user.event_detection.simhash_hamming_max

    rows = await session.execute(
        select(TechItem.simhash)
        .join(EventDocument, EventDocument.tech_item_id == TechItem.id)
        .where(EventDocument.event_id == event_id)
    )
    stored = [value for (value,) in rows.all()]
    return fold_near_duplicates(
        [from_storage(value) if value is not None else None for value in stored],
        max_distance=max_distance,
    )
