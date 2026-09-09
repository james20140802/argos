"""recluster 오케스트레이션 통합 검증 (ARG-281).

여러 sub를 가로지르는 것들을 여기 한 곳에서 본다: (1) 배선이 실제로 후보를
만들어 낸다 — 합칠 쪽도, 가를 쪽도, (2) 같은 기간을 두 번 돌리면 결과가 같다,
(3) 돌리고 나도 DB가 그대로다. DB나 그래프 라이브러리가 없으면 통째로 skip한다.

**(1)이 없으면 나머지가 공허하게 통과한다.** `first == second`도 `before ==
after`도 결과가 통째로 비어 있으면 그냥 참이다. 예컨대 `recluster.py`가
`event_links`를 `doc.tech_item_id`가 아니라 `doc.event_ids[0]`로 키잡으면
후보가 전부 사라지는데, 그래도 두 단언은 통과하고 CLI는 "합칠 후보: 없음"을
찍는다 — 사용자는 코퍼스가 깨끗하다고 읽는다. 그래서 아래 두 테스트는 후보의
**내용**을 못 박는다.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.recluster import recluster_period
from argos.config import settings
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable
from tests.conftest import requires_graph_libs

pytestmark = requires_graph_libs

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-281-recluster-e2e-test.example.com/"
_DIM = 768


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-281 recluster e2e test "
            "(start the Docker DB to run it)"
        )


@pytest.fixture(scope="module")
def session_factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    yield async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def clean(session_factory):
    async def _wipe():
        async with session_factory() as session:
            ids = (
                await session.execute(
                    select(TechItem.id).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
                )
            ).scalars().all()
            if ids:
                await session.execute(
                    delete(EventDocument).where(EventDocument.tech_item_id.in_(ids))
                )
                await session.execute(delete(TechItem).where(TechItem.id.in_(ids)))
            await session.execute(
                delete(TechEvent).where(TechEvent.title.like("ARG-281 %"))
            )
            await session.commit()

    await _wipe()
    yield
    await _wipe()


def _embedding(seed: float) -> list[float]:
    vec = [0.0] * _DIM
    vec[0] = 1.0
    vec[1] = seed
    return vec


def _directed_embedding(theta: float) -> list[float]:
    """방향을 각도로 직접 잡는 단위벡터 — 코사인 = cos(θ₁-θ₂)."""
    vec = [0.0] * _DIM
    vec[0] = math.cos(theta)
    vec[1] = math.sin(theta)
    return vec


@pytest.mark.asyncio
async def test_two_runs_agree_and_the_database_is_untouched(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    doc_ids = []
    async with session_factory() as session:
        # occurred_at은 NOT NULL이고 서버 기본값도 없다.
        event_a = TechEvent(title="ARG-281 event a", occurred_at=base)
        event_b = TechEvent(title="ARG-281 event b", occurred_at=base)
        session.add_all([event_a, event_b])
        await session.flush()
        for index in range(4):
            item = TechItem(
                title=f"ARG-281 doc{index}",
                source_url=f"{_URL_PREFIX}doc{index}",
                raw_content="x",
                summary="shared launch benchmark release",
                category=CategoryType.MAINSTREAM,
                published_at=base + timedelta(hours=index),
                embedding=_embedding(0.001 * index),
            )
            session.add(item)
            await session.flush()
            doc_ids.append(item.id)
            session.add(
                EventDocument(
                    event_id=event_a.id if index < 2 else event_b.id,
                    tech_item_id=item.id,
                )
            )
        await session.commit()
        expected_pair = tuple(sorted([event_a.id, event_b.id]))

    async with session_factory() as session:
        before = (
            (await session.execute(select(func.count(TechEvent.id)))).scalar(),
            (await session.execute(select(func.count(EventDocument.event_id)))).scalar(),
        )

    async with session_factory() as session:
        first = await recluster_period(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )
    async with session_factory() as session:
        second = await recluster_period(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    # 배선이 실제로 후보를 만들어 내는지부터 못 박는다 — 빈 결과로도 통과하는
    # 단언만 두면 `event_links` 키를 잘못 잡는 회귀를 잡지 못한다.
    # 실측(2026-09-10): 문서 4건이 한 커뮤니티로 묶이고 합칠 후보 1건이 나온다.
    assert [merge.event_ids for merge in first.merges] == [expected_pair]
    assert first.merges[0].evidence_document_ids == tuple(sorted(doc_ids))
    assert len(first.merges[0].evidence_document_ids) == 4
    assert first.splits == ()  # 두 사건이 한 덩어리이므로 가를 이유가 없다

    assert first == second  # AC: 같은 입력에 같은 결과

    async with session_factory() as session:
        after = (
            (await session.execute(select(func.count(TechEvent.id)))).scalar(),
            (await session.execute(select(func.count(EventDocument.event_id)))).scalar(),
        )
    assert before == after  # AC: DB가 조금도 변하지 않는다


@pytest.mark.asyncio
async def test_one_event_scattered_over_two_communities_becomes_a_split(
    session_factory, clean
):
    """가를 후보 쪽 배선도 실제로 도는지 본다 — 합칠 쪽과 분기가 다르다.

    한 사건에 문서 넷을 걸되, 임베딩 방향을 두 무리로 갈라 놓는다. 무리 안은
    코사인 1.0(가중치 ≈0.75)이라 간선이 되고, 무리 사이는 1.2rad 벌어져
    코사인 ≈0.36(가중치 ≈0.35)이라 간선이 못 된다. 그러면 커뮤니티가 둘로
    갈리고, 그 사건이 두 조각으로 보고돼야 한다.
    """
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    cluster_x, cluster_y = [], []
    async with session_factory() as session:
        event = TechEvent(title="ARG-281 event split", occurred_at=base)
        session.add(event)
        await session.flush()
        # summary도 무리별로 갈라 둔다 — 키워드 항까지 같으면 무리 사이
        # 가중치가 임계값 쪽으로 올라온다.
        plan = [
            ("x0", 0.0, "alpha compiler benchmark"),
            ("x1", 0.0, "alpha compiler benchmark"),
            ("y0", 1.2, "beta datacenter rollout"),
            ("y1", 1.2, "beta datacenter rollout"),
        ]
        for index, (slug, theta, summary) in enumerate(plan):
            item = TechItem(
                title=f"ARG-281 split {slug}",
                source_url=f"{_URL_PREFIX}split-{slug}",
                raw_content="x",
                summary=summary,
                category=CategoryType.MAINSTREAM,
                published_at=base + timedelta(hours=index),
                embedding=_directed_embedding(theta),
            )
            session.add(item)
            await session.flush()
            (cluster_x if theta == 0.0 else cluster_y).append(item.id)
            session.add(EventDocument(event_id=event.id, tech_item_id=item.id))
        await session.commit()
        event_id = event.id

    async with session_factory() as session:
        candidates = await recluster_period(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    assert candidates.merges == ()  # 사건이 하나뿐이라 합칠 쌍이 없다
    assert [split.event_id for split in candidates.splits] == [event_id]
    groups = {frozenset(group) for group in candidates.splits[0].groups}
    assert groups == {frozenset(cluster_x), frozenset(cluster_y)}
