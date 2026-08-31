"""시간 창 안 코사인 상위 K 이웃 조회 — ARG-265.

새 문서 하나를 사건에 온라인으로 배정하려면 "이 문서와 비슷하고, 시간상
가까운 문서들"이 먼저 필요하다. 이 모듈은 그 후보를 구하는 유일한 창구다.

**시간 창이 무엇을 줄이고 무엇을 줄이지 않는가:** 창은 *점수를 매길 후보
수*를 묶는다 — 7일 60건, 14일 140건, 30일 268건(2026-08-23, 코퍼스 1071건,
한쪽 창 기준 실측; 지금은 ``at`` 앞뒤 양쪽이라 상한은 그 두 배)이라 정렬과
뒤이은 가중치 계산이 작게 유지된다. 하지만 **스캔 자체는 줄이지
못한다.** 필터가 컬럼이 아니라 ``COALESCE(published_at, created_at)`` 위에
걸려 있어 기존 ``ix_tech_items_published_at``을 탈 수 없고, 플래너는 문서
하나를 배정할 때마다 tech_items를 통째로 순차 스캔한다 — 실측(2026-08-26,
코퍼스 1091건): ``Seq Scan``, ``Rows Removed by Filter: 979``,
``shared hit=942``, 7~8ms. 즉 비용은 창 안에 몇 건이 들어오느냐가 아니라
**코퍼스 전체 크기**를 따라 자란다. 예전 주석은 "창이 전수 스캔을 막는
수단"이라고 적었지만 그건 사실이 아니었다.

고칠 방법은 인덱스를 컬럼이 아니라 **표현식**에 거는 것이다:
``COALESCE(published_at, created_at)``에 부분 인덱스(``WHERE embedding IS NOT
NULL``)를 얹으면 정확한 정렬을 그대로 둔 채 인덱스를 다시 타게 만들 수 있다.
여기서 하지 않는 건 이유가 있다 — 이 브랜치에 허용된 일회성 Alembic 예외는
이미 있는 리비전이 가져갔고, 표현식 인덱스는 따로 리뷰받아야 할 별개 변경이다.
의도적으로 후속으로 미룬다.

**그렇다고 ANN 인덱스가 답인 것은 아니다:** 위 스캔 실측이 어떻게 바뀌든 이
판단은 따로 선다. HNSW/IVFFlat을 얹으면 플래너가 시간 창 필터를 무시하고
인덱스부터 타 버려 오히려 순서가 흔들리고, 매 크롤마다 인덱스 삽입 비용을
물며, 강제로 쓰면 근사 정렬이 되어 "같은 입력이면 같은 배정"이라는 결정성
기준이 깨진다. 그래서 이 모듈에는 인덱스 생성 코드도, 인덱스를 힌트하는
코드도 없다.

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
from argos.services.event_resolution import resolve_event

_CANDIDATE_SQL = text(
    """
    SELECT id, embedding, summary, digest,
           COALESCE(published_at, created_at) AS occurred_at
    FROM tech_items
    WHERE embedding IS NOT NULL
      AND COALESCE(published_at, created_at) >= :window_start
      AND COALESCE(published_at, created_at) <= :window_end
      AND EXISTS (
          SELECT 1 FROM event_documents ed
          WHERE ed.tech_item_id = tech_items.id
      )
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


def as_vector(value: object) -> tuple[float, ...] | None:
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
    읽는다. 시간 창은 *at* **양쪽**으로 각각 ``window_days``만큼이다 — 늦게
    크롤돼 ``published_at``이 과거인 기사도, 이미 저장된 더 최신 기사를
    이웃으로 봐야 같은 사건에 붙는다 (간선의 시간감쇠도 절대 시간차라 창만
    한쪽이면 서로 어긋난다). 창 밖이거나 임베딩이 없는 문서는 제외된다.

    **사건에 속하지 않은 문서도 제외된다** — LIMIT 앞에서. 소비자는 이웃이
    이미 속한 사건으로만 표를 모으므로, 링크 없는 문서(레거시 코퍼스,
    배정 실패분)가 상위 K 슬롯을 차지하면 정작 같은 사건의 이웃이 밀려나
    중복 사건이 생긴다. 후보가 0건이면 빈 목록을 반환한다 — 예외를 던지지
    않는다.

    반환 순서는 코사인 거리 오름차순이고, 동점은 ``tech_items.id`` 오름차순
    으로 깨진다 (결정성 — 같은 입력은 항상 같은 순서). 이웃의 ``event_ids``는
    툼스톤 체인을 생존 사건까지 해석한 뒤 중복을 접은, id 오름차순 튜플이다.
    """
    config = settings.user.event_detection
    if window_days is None:
        window_days = config.window_days
    if limit is None:
        limit = config.candidate_k

    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    window_start = at - timedelta(days=window_days)
    window_end = at + timedelta(days=window_days)

    rows = (
        await session.execute(
            _CANDIDATE_SQL,
            {
                "window_start": window_start,
                "window_end": window_end,
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
    # 링크가 흡수된(tombstoned) 사건을 가리키면 생존 사건으로 해석한다 —
    # 옛 id로 표를 모으면 choose_event가 툼스톤을 골라 새 근거가 산 사건
    # 대신 툼스톤에 쌓인다. 해석은 "모든 사건 조회"의 공용 불변식이다
    # (services/event_resolution docstring). 같은 생존자로 접히는 중복은
    # 한 번만 남긴다 — 두 번 실으면 그 이웃의 표가 두 배로 계산된다.
    resolved_cache: dict[uuid.UUID, uuid.UUID] = {}
    raw_events_by_item: dict[uuid.UUID, list[uuid.UUID]] = {}
    for tech_item_id, event_id in event_rows.all():
        if event_id not in resolved_cache:
            resolved_cache[event_id] = await resolve_event(session, event_id)
        resolved = resolved_cache[event_id]
        bucket = raw_events_by_item.setdefault(tech_item_id, [])
        if resolved not in bucket:
            bucket.append(resolved)
    events_by_item = {
        item_id: sorted(bucket) for item_id, bucket in raw_events_by_item.items()
    }

    names_by_item = await names_for_documents(session, ids)

    candidates: list[CandidateNeighbor] = []
    for row in rows:
        keyword_source = row.summary or row.digest
        features = DocumentFeatures(
            embedding=as_vector(row.embedding),
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
