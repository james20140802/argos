"""기간 단위 재군집 입력 조회 — ARG-278. 읽기 전용이다.

야간 재군집은 문서 하나가 아니라 **기간 전체**를 다시 묶는다. 그래서 온라인
배정의 `fetch_candidates`를 문서마다 N번 부르면 왕복도 N번, 스캔도 N × 코퍼스가
된다. 여기서는 LATERAL 조인 한 번으로 "기간 안 문서 × 각자의 시간 창 안 상위 K
이웃"을 한 방에 읽는다.

**1단계와 무엇이 같고 무엇이 다른가.** 시간 창(`±window_days`)도, 정확 정렬
(`ORDER BY embedding <=> :emb, id LIMIT k`)도, 동점을 id 오름차순으로 깨는
것도 그대로다 — 결정성 기준이 같아야 낮의 배정과 밤의 교정이 같은 판단을
한다. 다른 건 하나뿐이다: **`EXISTS event_documents` 필터를 걸지 않는다.**
성공 장면이 "기간의 문서 **전체**"라고 말하기 때문에, 아직 어떤 사건에도 붙지
않은 문서도 재군집 대상이다.

**ANN 인덱스는 쓰지 않는다** — `event_candidates` 모듈 docstring의 이유가 그대로
적용된다. 근사 정렬은 "같은 입력이면 같은 결과"를 깬다.

이웃 쌍은 `(작은 id, 큰 id)`로 정규화해 중복을 접는다. A가 B를 이웃으로 꼽고
B도 A를 꼽으면 같은 간선이므로 두 번 실으면 그래프에서 무게가 두 배가 된다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_store import names_for_documents
from argos.brain.event_candidates import as_vector, keywords_of
from argos.brain.event_scoring import DocumentFeatures
from argos.config import settings
from argos.models.event_document import EventDocument
from argos.services.event_resolution import resolve_event

_PERIOD_DOCS_SQL = text(
    """
    SELECT id, embedding, summary, digest,
           COALESCE(published_at, created_at) AS occurred_at
    FROM tech_items
    WHERE COALESCE(published_at, created_at) >= :start
      AND COALESCE(published_at, created_at) <= :end
    ORDER BY id
    """
)
"""기간 안 문서 전부. 임베딩이 없는 문서도 포함한다 — 이웃 후보는 못 되지만
재군집 대상(그리고 결과 커뮤니티의 싱글턴)으로는 남아야 한다."""

_NEIGHBOR_PAIRS_SQL = text(
    """
    WITH period AS (
        SELECT id, embedding, COALESCE(published_at, created_at) AS occurred_at
        FROM tech_items
        WHERE embedding IS NOT NULL
          AND COALESCE(published_at, created_at) >= :start
          AND COALESCE(published_at, created_at) <= :end
    )
    SELECT p.id AS left_id, n.id AS right_id
    FROM period p
    CROSS JOIN LATERAL (
        SELECT t.id
        FROM tech_items t
        WHERE t.embedding IS NOT NULL
          AND t.id <> p.id
          AND COALESCE(t.published_at, t.created_at)
              >= p.occurred_at - make_interval(days => :window_days)
          AND COALESCE(t.published_at, t.created_at)
              <= p.occurred_at + make_interval(days => :window_days)
        ORDER BY t.embedding <=> p.embedding, t.id
        LIMIT :limit
    ) n
    ORDER BY p.id, n.id
    """
)
"""문서당 시간 창 안 상위 K 이웃. LATERAL이라 왕복은 한 번이고, 쌍의 수는
문서 수 × K로 묶인다 — 문서 수의 제곱으로 자라지 않는다."""


@dataclass(frozen=True)
class ReclusterDocument:
    """재군집 대상 문서 한 건 — 피처와 **현재** 사건 링크(생존 사건 기준)."""

    tech_item_id: uuid.UUID
    features: DocumentFeatures
    event_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class NeighborPair:
    """점수를 매겨 볼 문서 쌍. 항상 `left_id < right_id`로 정규화돼 있다."""

    left_id: uuid.UUID
    right_id: uuid.UUID


@dataclass(frozen=True)
class ReclusterInput:
    """한 기간의 재군집 입력 전체. 순수 코어(T3/T4)는 이것만 받는다."""

    documents: tuple[ReclusterDocument, ...]
    neighbor_pairs: tuple[NeighborPair, ...]


async def fetch_period_input(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    window_days: float | None = None,
    limit: int | None = None,
) -> ReclusterInput:
    """`start`~`end`(양끝 포함) 기간의 재군집 입력을 읽는다. 쓰기는 없다.

    `window_days`/`limit`이 None이면 `event_detection` config에서 읽는다 —
    1단계 배정과 같은 값을 쓰기 위해서다.

    문서는 id 오름차순, 이웃 쌍은 `(left_id, right_id)` 오름차순으로 정렬돼
    돌아온다. 같은 입력이면 항상 같은 순서다.
    """
    config = settings.user.event_detection
    if window_days is None:
        window_days = config.window_days
    if limit is None:
        limit = config.candidate_k

    doc_rows = (
        await session.execute(_PERIOD_DOCS_SQL, {"start": start, "end": end})
    ).all()
    if not doc_rows:
        return ReclusterInput(documents=(), neighbor_pairs=())

    ids = [row.id for row in doc_rows]

    event_rows = await session.execute(
        select(EventDocument.tech_item_id, EventDocument.event_id)
        .where(EventDocument.tech_item_id.in_(ids))
        .order_by(EventDocument.event_id)
    )
    # 툼스톤 체인은 생존 사건까지 해석한다 — "모든 사건 조회"의 공용 불변식
    # (services/event_resolution docstring). 같은 생존자로 접히는 중복은 한
    # 번만 남긴다.
    resolved_cache: dict[uuid.UUID, uuid.UUID] = {}
    raw_events_by_item: dict[uuid.UUID, list[uuid.UUID]] = {}
    for tech_item_id, event_id in event_rows.all():
        if event_id not in resolved_cache:
            resolved_cache[event_id] = await resolve_event(session, event_id)
        resolved = resolved_cache[event_id]
        bucket = raw_events_by_item.setdefault(tech_item_id, [])
        if resolved not in bucket:
            bucket.append(resolved)

    names_by_item = await names_for_documents(session, ids)

    documents = tuple(
        ReclusterDocument(
            tech_item_id=row.id,
            features=DocumentFeatures(
                embedding=as_vector(row.embedding),
                names=names_by_item.get(row.id, frozenset()),
                at=row.occurred_at,
                keywords=keywords_of(row.summary or row.digest),
            ),
            event_ids=tuple(sorted(raw_events_by_item.get(row.id, []))),
        )
        for row in doc_rows
    )

    pair_rows = (
        await session.execute(
            _NEIGHBOR_PAIRS_SQL,
            {
                "start": start,
                "end": end,
                "window_days": float(window_days),
                "limit": limit,
            },
        )
    ).all()

    # 방향을 정규화해 접는다: A→B와 B→A는 같은 간선이다. 기간 밖 이웃은
    # 재군집 대상이 아니므로 버린다 — 창은 기간 경계를 넘어 보지만, 그래프의
    # 노드는 기간 안 문서로 한정한다.
    in_period = {row.id for row in doc_rows}
    pairs = {
        (min(row.left_id, row.right_id), max(row.left_id, row.right_id))
        for row in pair_rows
        if row.right_id in in_period
    }
    neighbor_pairs = tuple(
        NeighborPair(left_id=left, right_id=right) for left, right in sorted(pairs)
    )

    return ReclusterInput(documents=documents, neighbor_pairs=neighbor_pairs)
