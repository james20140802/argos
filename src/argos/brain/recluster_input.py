"""기간 단위 재군집 입력 조회 — ARG-278. 읽기 전용이다.

야간 재군집은 문서 하나가 아니라 **기간 전체**를 다시 묶는다. 그래서 온라인
배정의 `fetch_candidates`를 문서마다 N번 부르면 왕복도 N번, 스캔도 N × 코퍼스가
된다. 여기서는 LATERAL 조인 한 번으로 "기간 안 문서 × 각자의 시간 창 안 상위 K
이웃"을 한 방에 읽는다.

**1단계와 무엇이 같고 무엇이 다른가.** 시간 창(`±window_days`)도, 정확 정렬
(`ORDER BY embedding <=> :emb, id LIMIT k`)도, 동점을 id 오름차순으로 깨는
것도 그대로다 — 같은 정렬 규칙을 써야 "같은 입력이면 같은 결과"가 두 경로에서
같은 뜻이 된다. 다른 건 둘이다.

1. **`EXISTS event_documents` 필터를 걸지 않는다.** 성공 장면이 "기간의 문서
   **전체**"라고 말하기 때문에, 아직 어떤 사건에도 붙지 않은 문서도 재군집
   대상이다.
2. **상위 K를 코퍼스 전체가 아니라 기간 안에서 고른다.** 1단계는 새 문서
   하나를 기존 코퍼스에 붙이는 일이라 창 안 아무나 이웃이 될 수 있지만,
   재군집의 그래프는 노드가 기간 안 문서뿐이다(`_NEIGHBOR_PAIRS_SQL` docstring).

**ANN 인덱스는 쓰지 않는다** — `event_candidates` 모듈 docstring의 이유가 그대로
적용된다. 근사 정렬은 "같은 입력이면 같은 결과"를 깬다.

이웃 쌍은 `(작은 id, 큰 id)`로 정규화해 중복을 접는다. A가 B를 이웃으로 꼽고
B도 A를 꼽으면 같은 간선이므로 두 번 실으면 그래프에서 무게가 두 배가 된다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import bindparam, select, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PGUuid
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_store import names_for_documents
from argos.brain.event_candidates import as_vector, keywords_of
from argos.brain.event_scoring import DocumentFeatures
from argos.config import settings
from argos.models.event_document import EventDocument
from argos.services.event_resolution import resolve_events

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
          AND id = ANY(:ids)
    )
    SELECT p.id AS left_id, n.id AS right_id
    FROM period p
    CROSS JOIN LATERAL (
        SELECT t.id
        FROM period t
        WHERE t.id <> p.id
          AND t.occurred_at >= p.occurred_at - :window_days * INTERVAL '1 day'
          AND t.occurred_at <= p.occurred_at + :window_days * INTERVAL '1 day'
        ORDER BY t.embedding <=> p.embedding, t.id
        LIMIT :limit
    ) n
    ORDER BY p.id, n.id
    """
).bindparams(bindparam("ids", type_=ARRAY(PGUuid(as_uuid=True))))
"""문서당 시간 창 안 상위 K 이웃. LATERAL이라 왕복은 한 번이고, 쌍의 수는
문서 수 × K로 묶인다 — 문서 수의 제곱으로 자라지 않는다.

**기간을 날짜로 다시 긋지 않고 앞서 읽은 문서 id로 묶는다.** 기본 격리
수준(READ COMMITTED)에서 이 조회는 문서 조회와 다른 스냅샷을 본다. 날짜로
다시 그으면 그 사이 `argos run`이 커밋한 문서까지 순위에 끼어들고, 그 문서는
`documents`에 없으니 코어가 쌍을 버린다 — 결국 **원래 있던 간선만 사라져**
멀쩡한 사건이 "가를 후보"로 잡힌다. 세션의 격리 수준을 올리지 않고 id로 묶는
쪽을 고른 건, 이 함수가 호출자의 세션을 빌려 쓰는 처지라 트랜잭션 semantics를
바꾸는 부작용을 남기면 안 되기 때문이다. 나머지 후속 조회(사건 링크·이름)는
이미 같은 id 목록으로 묶여 있다.

**순위는 기간 안에서 매긴다** — 안쪽 FROM이 `tech_items`가 아니라 `period`다.
그래프의 노드는 어차피 기간 안 문서뿐이라 기간 밖 이웃은 뽑아 봐야 버려지는데,
LIMIT 자리를 먼저 차지하면 정작 기간 안 이웃이 K등 밖으로 밀린다. 창이 빽빽한
코퍼스에서는 그렇게 사라진 간선 때문에 멀쩡한 사건이 "가를 후보"로 잘못 잡히고,
결과가 기간 경계를 어디에 그었느냐에 따라 달라진다.
`period`가 이미 `embedding IS NOT NULL`과 `occurred_at`을 들고 있어서 안쪽에서
다시 걸 조건도 없다. 정렬(`ORDER BY t.embedding <=> p.embedding, t.id`)은 그대로다
— 그게 결정성 보장이다."""


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
    #
    # 사건당 한 번씩 묻지 않고 배치로 가는 이유: 기간 전체 재군집은 사건이
    # 수백 개일 수 있고(특히 1단계 배정이 문서당 사건 하나를 만들어 둔 기간),
    # 그러면 그래프 계산 전에 직렬 왕복만 수백 번이다. 나머지 입력을 전부 한
    # 방 조회로 읽어 온 보람이 사라진다. 배치 판은 왕복이 사건 수가 아니라
    # 툼스톤 체인 깊이를 따르고, 답은 하나씩 부른 것과 같다.
    links = event_rows.all()
    resolved_by_event = await resolve_events(
        session, {event_id for _, event_id in links}
    )
    raw_events_by_item: dict[uuid.UUID, list[uuid.UUID]] = {}
    for tech_item_id, event_id in links:
        resolved = resolved_by_event[event_id]
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
                "ids": ids,
                "window_days": float(window_days),
                "limit": limit,
            },
        )
    ).all()

    # 방향을 정규화해 접는다: A→B와 B→A는 같은 간선이다. 기간 밖 이웃을
    # 파이썬에서 걸러 내던 코드는 없앴다 — 이제 SQL이 `period` 안에서만 이웃을
    # 뽑으므로 여기 오는 쌍의 양끝은 전부 기간 안 문서다(그리고 걸러 내는 대신
    # 애초에 뽑지 않으니 상위 K 자리를 기간 밖 문서에 뺏기지도 않는다).
    pairs = {
        (min(row.left_id, row.right_id), max(row.left_id, row.right_id))
        for row in pair_rows
    }
    neighbor_pairs = tuple(
        NeighborPair(left_id=left, right_id=right) for left, right in sorted(pairs)
    )

    return ReclusterInput(documents=documents, neighbor_pairs=neighbor_pairs)
