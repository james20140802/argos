"""ARG-240 DB integration test — 툼스톤 포인터가 삭제로 끊기지 않는지 본다.

병합은 삭제가 아니라 툼스톤이다(``TechEvent.merged_into_id``). 그래서 흡수해
간 사건(병합 대상)은 지워지면 안 되고, self-FK의 ``ondelete="RESTRICT"``가 DB
레벨에서 그걸 막는다.

그런데 **모델에 FK만 걸어두면 이 방어가 통째로 무력화된다.** ``AsyncSession``
으로 병합 대상을 지우면 SQLAlchemy의 기본 delete synchronization이 자식들의
``merged_into_id``를 먼저 ``NULL``로 밀어버리고, 그러면 Postgres는 참조하는
행을 아예 못 보므로 RESTRICT가 발동하지 않는다 — 삭제는 조용히 성공하고 옛
id는 더 이상 생존 사건으로 이어지지 않는다. ``merged_from`` 관계의
``passive_deletes="all"``이 그 사전 NULL 처리를 끄고 FK가 실제로 말하게 한다.

이 파일은 그 회귀를 잡는다. ``passive_deletes="all"``을 지우면 첫 테스트가
"삭제가 거부되지 않았다"로 깨진다.

Postgres가 없으면 통째로 skip한다 (release.yml에는 DB 서비스가 없다 —
CLAUDE.md "Release CI runs pytest with no DB" 참고).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.config import settings
from argos.models.tech_event import TechEvent
from tests.conftest import db_reachable as _db_reachable

# Captured at import time, same pattern as tests/test_cli_backfill_images_db.py.
_DB_URL: str = settings.database_url


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-240 tombstone DB "
            "integration test (start the Docker DB to run it)"
        )


@pytest.fixture
async def session_factory():
    """NullPool 기반 sessionmaker를 주고, 끝나면 이 파일이 만든 행을 지운다."""
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with factory() as session:
            # 자식(툼스톤)을 먼저 지워야 RESTRICT에 걸리지 않는다.
            await session.execute(
                delete(TechEvent).where(
                    TechEvent.title.like("ARG-240 tombstone test%"),
                    TechEvent.merged_into_id.isnot(None),
                )
            )
            await session.execute(
                delete(TechEvent).where(
                    TechEvent.title.like("ARG-240 tombstone test%")
                )
            )
            await session.commit()
        await engine.dispose()


async def _make_absorbed_pair(factory):
    """B(생존)와 그 안으로 흡수된 A를 만들고 (a_id, b_id)를 돌려준다."""
    async with factory() as session:
        survivor = TechEvent(
            title="ARG-240 tombstone test — survivor",
            occurred_at=datetime.now(timezone.utc),
        )
        session.add(survivor)
        await session.flush()
        absorbed = TechEvent(
            title="ARG-240 tombstone test — absorbed",
            occurred_at=datetime.now(timezone.utc),
            merged_into_id=survivor.id,
        )
        session.add(absorbed)
        await session.commit()
        return absorbed.id, survivor.id


@pytest.mark.asyncio
async def test_deleting_a_merge_target_is_refused_by_the_database(session_factory):
    """흡수해 간 사건은 ORM 경유 삭제도 DB가 거부한다."""
    absorbed_id, survivor_id = await _make_absorbed_pair(session_factory)

    async with session_factory() as session:
        survivor = await session.get(TechEvent, survivor_id)
        with pytest.raises(IntegrityError):
            await session.delete(survivor)
            await session.flush()
        await session.rollback()

    # 툼스톤 포인터는 그대로 살아 있어야 한다 — 옛 id가 계속 이어져야 하므로.
    async with session_factory() as session:
        pointer = await session.scalar(
            select(TechEvent.merged_into_id).where(TechEvent.id == absorbed_id)
        )
        assert pointer == survivor_id


@pytest.mark.asyncio
async def test_deleting_an_event_nobody_merged_into_still_works(session_factory):
    """RESTRICT가 과하게 막지는 않는지 — 흡수한 적 없는 사건은 지워진다."""
    async with session_factory() as session:
        lone = TechEvent(
            title="ARG-240 tombstone test — lone",
            occurred_at=datetime.now(timezone.utc),
        )
        session.add(lone)
        await session.commit()
        lone_id = lone.id

    async with session_factory() as session:
        target = await session.get(TechEvent, lone_id)
        await session.delete(target)
        await session.commit()

    async with session_factory() as session:
        assert await session.get(TechEvent, lone_id) is None
