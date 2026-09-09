"""recluster_input — 기간 단위 재군집 입력 조회의 DB 통합 테스트 (ARG-278).

패턴은 `tests/brain/test_event_candidates_db.py`와 같다: 모듈 스코프
session_factory(NullPool) + 이 모듈이 만든 행만 정리 + Postgres 없으면 skip.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.recluster_input import fetch_period_input
from argos.config import settings
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-278-recluster-input-test.example.com/"
_DIM = 768


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-278 recluster_input DB "
            "integration test (start the Docker DB to run it)"
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
                delete(TechEvent).where(TechEvent.title.like("ARG-278 %"))
            )
            await session.commit()

    await _wipe()
    yield
    await _wipe()


def _embedding(seed: float) -> list[float]:
    """첫 두 성분만 다른 단위 벡터 — 코사인 거리를 눈으로 통제하기 위해."""
    vec = [0.0] * _DIM
    vec[0] = 1.0
    vec[1] = seed
    return vec


async def _make_item(session, *, slug: str, at: datetime, seed: float) -> uuid.UUID:
    item = TechItem(
        title=f"ARG-278 {slug}",
        source_url=f"{_URL_PREFIX}{slug}",
        raw_content="x",
        summary=f"summary for {slug}",
        category=CategoryType.MAINSTREAM,
        published_at=at,
        embedding=_embedding(seed),
    )
    session.add(item)
    await session.flush()
    return item.id


async def _make_event(
    session, *, title: str, at: datetime, merged_into=None
) -> uuid.UUID:
    # occurred_at은 NOT NULL이고 서버 기본값도 없다 — 반드시 채워 준다.
    event = TechEvent(
        title=f"ARG-278 {title}", occurred_at=at, merged_into_id=merged_into
    )
    session.add(event)
    await session.flush()
    return event.id


@pytest.mark.asyncio
async def test_period_includes_documents_with_no_event_link(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    async with session_factory() as session:
        linked = await _make_item(session, slug="linked", at=base, seed=0.01)
        orphan = await _make_item(session, slug="orphan", at=base, seed=0.02)
        event_id = await _make_event(session, title="e1", at=base)
        session.add(EventDocument(event_id=event_id, tech_item_id=linked))
        await session.commit()

    async with session_factory() as session:
        result = await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    ids = {doc.tech_item_id for doc in result.documents}
    assert linked in ids
    assert orphan in ids  # AC: 사건에 안 붙은 문서도 입력에 들어온다
    by_id = {doc.tech_item_id: doc for doc in result.documents}
    assert by_id[orphan].event_ids == ()
    assert by_id[linked].event_ids == (event_id,)


@pytest.mark.asyncio
async def test_documents_outside_the_period_are_excluded(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    async with session_factory() as session:
        inside = await _make_item(session, slug="inside", at=base, seed=0.01)
        await _make_item(session, slug="outside", at=base + timedelta(days=30), seed=0.02)
        await session.commit()

    async with session_factory() as session:
        result = await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    assert [doc.tech_item_id for doc in result.documents] == [inside]


@pytest.mark.asyncio
async def test_two_identical_queries_agree_down_to_the_order(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    async with session_factory() as session:
        for index in range(6):
            await _make_item(
                session,
                slug=f"det{index}",
                at=base + timedelta(hours=index),
                seed=0.01 * index,
            )
        await session.commit()

    async with session_factory() as session:
        first = await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )
    async with session_factory() as session:
        second = await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    assert [d.tech_item_id for d in first.documents] == [
        d.tech_item_id for d in second.documents
    ]
    assert first.neighbor_pairs == second.neighbor_pairs


@pytest.mark.asyncio
async def test_pairs_are_capped_per_document_not_quadratic(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    doc_count = 8
    async with session_factory() as session:
        for index in range(doc_count):
            await _make_item(
                session,
                slug=f"cap{index}",
                at=base + timedelta(hours=index),
                seed=0.001 * index,
            )
        await session.commit()

    async with session_factory() as session:
        result = await fetch_period_input(
            session,
            start=base - timedelta(days=1),
            end=base + timedelta(days=1),
            limit=2,
        )

    # 문서당 상위 2개 → 상한은 doc_count * 2 (중복 접기 전). 완전 그래프인
    # doc_count*(doc_count-1)/2 = 28보다 확실히 작아야 한다.
    assert len(result.neighbor_pairs) <= doc_count * 2
    assert len(result.neighbor_pairs) < doc_count * (doc_count - 1) // 2


@pytest.mark.asyncio
async def test_pairs_are_normalized_and_deduplicated(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    async with session_factory() as session:
        await _make_item(session, slug="p1", at=base, seed=0.01)
        await _make_item(session, slug="p2", at=base + timedelta(hours=1), seed=0.011)
        await session.commit()

    async with session_factory() as session:
        result = await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    for pair in result.neighbor_pairs:
        assert pair.left_id < pair.right_id  # 정규화된 방향
    assert len(set(result.neighbor_pairs)) == len(result.neighbor_pairs)


@pytest.mark.asyncio
async def test_tombstoned_event_links_resolve_to_the_survivor(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    async with session_factory() as session:
        survivor = await _make_event(session, title="survivor", at=base)
        absorbed = await _make_event(
            session, title="absorbed", at=base, merged_into=survivor
        )
        item = await _make_item(session, slug="tomb", at=base, seed=0.01)
        session.add(EventDocument(event_id=absorbed, tech_item_id=item))
        await session.commit()

    async with session_factory() as session:
        result = await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    by_id = {doc.tech_item_id: doc for doc in result.documents}
    assert by_id[item].event_ids == (survivor,)  # 툼스톤이 아니라 생존자


@pytest.mark.asyncio
async def test_query_writes_nothing(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
    async with session_factory() as session:
        item = await _make_item(session, slug="ro", at=base, seed=0.01)
        event_id = await _make_event(session, title="ro-event", at=base)
        session.add(EventDocument(event_id=event_id, tech_item_id=item))
        await session.commit()

    async with session_factory() as session:
        before_events = (await session.execute(select(func.count(TechEvent.id)))).scalar()
        before_links = (
            await session.execute(select(func.count(EventDocument.event_id)))
        ).scalar()

    async with session_factory() as session:
        await fetch_period_input(
            session, start=base - timedelta(days=1), end=base + timedelta(days=1)
        )

    async with session_factory() as session:
        after_events = (await session.execute(select(func.count(TechEvent.id)))).scalar()
        after_links = (
            await session.execute(select(func.count(EventDocument.event_id)))
        ).scalar()

    assert (before_events, before_links) == (after_events, after_links)
