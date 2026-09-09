"""recluster 오케스트레이션 통합 검증 (ARG-281).

여러 sub를 가로지르는 두 가지를 여기 한 곳에서 본다: (1) 같은 기간을 두 번
돌리면 결과가 같다, (2) 돌리고 나도 DB가 그대로다. DB나 그래프 라이브러리가
없으면 통째로 skip한다.
"""
from __future__ import annotations

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


@pytest.mark.asyncio
async def test_two_runs_agree_and_the_database_is_untouched(session_factory, clean):
    base = datetime(2026, 8, 10, tzinfo=timezone.utc)
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
            session.add(
                EventDocument(
                    event_id=event_a.id if index < 2 else event_b.id,
                    tech_item_id=item.id,
                )
            )
        await session.commit()

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

    assert first == second  # AC: 같은 입력에 같은 결과

    async with session_factory() as session:
        after = (
            (await session.execute(select(func.count(TechEvent.id)))).scalar(),
            (await session.execute(select(func.count(EventDocument.event_id)))).scalar(),
        )
    assert before == after  # AC: DB가 조금도 변하지 않는다
