"""기간 재군집 오케스트레이션 — ARG-281. 읽기 전용이다.

조회(T2) → (애매한 짝 확인) → 커뮤니티 판정(T3) → 후보 도출(T4)을 잇는 얇은
층이다. 판단은 전부 아래 모듈들에 있고 여기에는 없다 — 그래야 조각을 각자 DB
없이(또는 DB만으로) 검증할 수 있다.

사건 링크는 조회 계층이 이미 생존 사건으로 해석해 준 것을 그대로 넘긴다.
후보 도출은 순수 함수라 DB를 볼 수 없고, 툼스톤 해석에는 DB가 필요하기 때문이다.

애매한 짝 확인(ARG-307)은 `pair_check_enabled`일 때만, 조회 뒤 커뮤니티 판정 앞에서
돈다. LLM 호출은 이 비동기 층에만 있고 판정 코어는 순수·동기 그대로다. 확인이
실패해도(Ollama 꺼짐 등) 거부 집합이 비어 있을 뿐이라 교정은 이 기능이 없는 것과
같게 진행된다.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.pair_check import (
    PairVerdictCache,
    check_ambiguous_pairs,
    default_cache_path,
)
from argos.brain.recluster_candidates import ReclusterCandidates, derive_candidates
from argos.brain.recluster_core import detect_communities, score_pairs
from argos.brain.recluster_input import fetch_period_input
from argos.config import settings


async def recluster_period(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    pair_check_max_calls: int | None = None,
) -> ReclusterCandidates:
    """`start`~`end` 기간을 다시 묶어 교정 후보를 계산한다. 쓰기는 없다.

    Args:
        pair_check_max_calls: 이번 실행의 LLM 호출 상한. None이면 설정값.

    Raises:
        GraphLibsUnavailable: python-igraph/leidenalg 미설치.
    """
    config = settings.user.event_detection
    period = await fetch_period_input(session, start=start, end=end)
    if not period.documents:
        return ReclusterCandidates(merges=(), splits=())

    report = None
    if config.pair_check_enabled:
        cache_path = (
            Path(config.pair_check_cache_path).expanduser()
            if config.pair_check_cache_path
            else default_cache_path()
        )
        report = await check_ambiguous_pairs(
            session,
            period.documents,
            score_pairs(period.documents, period.neighbor_pairs, config=config),
            config=config,
            cache=PairVerdictCache.load(cache_path),
            max_calls=pair_check_max_calls,
        )

    communities = detect_communities(
        period.documents,
        period.neighbor_pairs,
        config=config,
        vetoed_pairs=report.vetoed if report is not None else frozenset(),
    )
    event_links = {
        doc.tech_item_id: doc.event_ids for doc in period.documents if doc.event_ids
    }
    candidates = derive_candidates(
        communities,
        event_links,
        capped_document_ids=period.capped_document_ids,
    )
    return replace(candidates, pair_check=report)
