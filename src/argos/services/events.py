"""사건에 매달린 근거 문서 조회.

쿼리 조립부(``build_evidence_documents_query``)와 실행부
(``list_evidence_documents``)를 나눠 둔 이유는, 조립부가 세션 없이 컴파일만으로
검증 가능해서 Postgres 없는 환경에서도 단위 테스트가 돌기 때문이다.

이 쿼리들은 넘겨받은 ``event_id``를 그대로 매치한다 — 툼스톤 체인은 따라가지
않는다. 사건이 병합돼도 흡수된 사건 아래 걸린 근거 문서 링크는 저절로 생존
사건 쪽으로 옮겨오지 않으므로, 호출자는 ``resolve_event()``로 미리 해석한 id를
넘겨야 한다. 링크를 생존 사건으로 옮기는 일은 병합 작성자(merge-writer, 후속
이슈)의 몫이며 이 이슈에서는 구현하지 않는다.
"""
from __future__ import annotations

import uuid

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.event_document import EventDocument
from argos.models.tech_item import TechItem


def build_evidence_documents_query(event_id: uuid.UUID) -> Select:
    """사건 id → 근거 문서(TechItem) 목록을 뽑는 SELECT를 조립한다.

    최신 문서가 앞에 오도록 ``published_at`` 내림차순으로 정렬하고,
    발행일을 모르는 문서는 뒤로 보낸다.
    """
    return (
        select(TechItem)
        .join(EventDocument, EventDocument.tech_item_id == TechItem.id)
        .where(EventDocument.event_id == event_id)
        .order_by(TechItem.published_at.desc().nulls_last())
    )


async def list_evidence_documents(
    session: AsyncSession,
    event_id: uuid.UUID,
) -> list[TechItem]:
    """사건의 근거 문서 목록을 반환한다. 근거가 없으면 빈 리스트."""
    result = await session.execute(build_evidence_documents_query(event_id))
    return list(result.scalars().all())
