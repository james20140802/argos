"""기간 재군집 오케스트레이션 — ARG-281. 읽기 전용이다.

조회(T2) → 커뮤니티 판정(T3) → 후보 도출(T4)을 잇는 얇은 층이다. 판단은 전부
아래 세 모듈에 있고 여기에는 없다 — 그래야 세 조각을 각자 DB 없이(또는 DB만으로)
검증할 수 있다.

사건 링크는 조회 계층이 이미 생존 사건으로 해석해 준 것을 그대로 넘긴다.
후보 도출은 순수 함수라 DB를 볼 수 없고, 툼스톤 해석에는 DB가 필요하기 때문이다.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.recluster_candidates import ReclusterCandidates, derive_candidates
from argos.brain.recluster_core import detect_communities
from argos.brain.recluster_input import fetch_period_input
from argos.config import settings


async def recluster_period(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
) -> ReclusterCandidates:
    """`start`~`end` 기간을 다시 묶어 교정 후보를 계산한다. 쓰기는 없다.

    Raises:
        GraphLibsUnavailable: python-igraph/leidenalg 미설치.
    """
    config = settings.user.event_detection
    period = await fetch_period_input(session, start=start, end=end)
    if not period.documents:
        return ReclusterCandidates(merges=(), splits=())

    communities = detect_communities(
        period.documents, period.neighbor_pairs, config=config
    )
    event_links = {
        doc.tech_item_id: doc.event_ids for doc in period.documents if doc.event_ids
    }
    return derive_candidates(communities, event_links)
