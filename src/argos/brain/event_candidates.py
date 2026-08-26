"""시간 창 안 코사인 상위 K 이웃 조회 — ARG-265.

새 문서 하나를 사건에 온라인으로 배정하려면 "이 문서와 비슷하고, 시간상
가까운 문서들"이 먼저 필요하다. 이 모듈은 그 후보를 구하는 유일한 창구다.

**왜 ANN 인덱스를 쓰지 않는가:** 시간 창이 전수 스캔을 막는 수단이다. 실측
(2026-08-23, 코퍼스 1071건) 기준 7일 창은 60건, 14일 창은 140건, 30일 창도
268건이라 인덱스 없는 순차 스캔 + 정렬로 수 ms 안에 끝난다. HNSW/IVFFlat을
얹으면 플래너가 시간 창 필터를 무시하고 인덱스부터 타 버려 오히려 순서가
흔들리고, 매 크롤마다 인덱스 삽입 비용을 물며, 강제로 쓰면 근사 정렬이 되어
"같은 입력이면 같은 배정"이라는 결정성 기준이 깨진다. 그래서 이 모듈에는
인덱스 생성 코드도, 인덱스를 힌트하는 코드도 없다.

**3-쿼리 구조인 이유:** 상위 K 행을 먼저 확정한 뒤 그 id 목록으로 사건 링크와
이름을 각각 한 번씩 더 읽는다. 단일 거대 조인보다 읽기 쉽고, K가 작아
(기본 25) 왕복 비용이 문제되지 않는다. 문서마다 도는 N+1은 피한다.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_store import names_for_documents
from argos.brain.event_scoring import DocumentFeatures
from argos.config import settings
from argos.models.event_document import EventDocument

_CANDIDATE_SQL = text(
    """
    SELECT id, embedding, summary, digest,
           COALESCE(published_at, created_at) AS occurred_at
    FROM tech_items
    WHERE embedding IS NOT NULL
      AND COALESCE(published_at, created_at) >= :window_start
      AND COALESCE(published_at, created_at) <= :at
      AND (CAST(:exclude_id AS uuid) IS NULL OR id <> CAST(:exclude_id AS uuid))
    ORDER BY embedding <=> CAST(:emb AS vector), id
    LIMIT :limit
    """
)

_WORD = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True)
class CandidateNeighbor:
    """이웃 문서 한 건 — 피처와 이미 속한 사건(들)을 함께 실어 나른다."""

    tech_item_id: uuid.UUID
    features: DocumentFeatures
    event_ids: tuple[uuid.UUID, ...]


def keywords_of(text_value: str | None) -> frozenset[str]:
    """요약(없으면 다이제스트)에서 뽑은 소문자 낱말 집합.

    새 추출기를 만들지 않는다 — 간선 가중치에서 키워드 항의 몫은 0.05라
    정교함이 값을 못 한다 (``event_scoring`` 참고).
    """
    if not text_value:
        return frozenset()
    return frozenset(match.group().casefold() for match in _WORD.finditer(text_value))


def _as_vector(value: object) -> tuple[float, ...] | None:
    """pgvector 임베딩을 ``tuple[float, ...]``로 접는다.

    드라이버 경로에 따라 리스트로도, 문자열(``"[0.1,0.2,...]"``)로도 돌아올
    수 있어 둘 다 받는다.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip("[]")
        if not stripped:
            return ()
        return tuple(float(x) for x in stripped.split(","))
    return tuple(float(x) for x in value)


async def fetch_candidates(
    session: AsyncSession,
    *,
    embedding: Sequence[float],
    at: datetime,
    exclude_id: uuid.UUID | None = None,
    window_days: float | None = None,
    limit: int | None = None,
) -> list[CandidateNeighbor]:
    """*embedding*/*at* 기준 시간 창 안의 코사인 상위 K 이웃을 반환한다.

    ``window_days``/``limit``이 ``None``이면 ``event_detection`` config에서
    읽는다. 창 밖이거나 임베딩이 없는 문서는 제외된다. 후보가 0건이면 빈
    목록을 반환한다 — 예외를 던지지 않는다.

    반환 순서는 코사인 거리 오름차순이고, 동점은 ``tech_items.id`` 오름차순
    으로 깨진다 (결정성 — 같은 입력은 항상 같은 순서).
    """
    config = settings.user.event_detection
    if window_days is None:
        window_days = config.window_days
    if limit is None:
        limit = config.candidate_k

    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    window_start = at - timedelta(days=window_days)

    rows = (
        await session.execute(
            _CANDIDATE_SQL,
            {
                "window_start": window_start,
                "at": at,
                "exclude_id": exclude_id,
                "emb": embedding_str,
                "limit": limit,
            },
        )
    ).all()
    if not rows:
        return []

    ids = [row.id for row in rows]

    event_rows = await session.execute(
        select(EventDocument.tech_item_id, EventDocument.event_id)
        .where(EventDocument.tech_item_id.in_(ids))
        .order_by(EventDocument.event_id)
    )
    events_by_item: dict[uuid.UUID, list[uuid.UUID]] = {}
    for tech_item_id, event_id in event_rows.all():
        events_by_item.setdefault(tech_item_id, []).append(event_id)

    names_by_item = await names_for_documents(session, ids)

    candidates: list[CandidateNeighbor] = []
    for row in rows:
        keyword_source = row.summary or row.digest
        features = DocumentFeatures(
            embedding=_as_vector(row.embedding),
            names=names_by_item.get(row.id, frozenset()),
            at=row.occurred_at,
            keywords=keywords_of(keyword_source),
        )
        candidates.append(
            CandidateNeighbor(
                tech_item_id=row.id,
                features=features,
                event_ids=tuple(events_by_item.get(row.id, [])),
            )
        )
    return candidates
