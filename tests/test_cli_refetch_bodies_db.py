"""ARG-309: `refetch-bodies` DB 왕복 — raw_content와 simhash만 바뀌는지 확인한다.

Postgres가 없으면(release.yml) 통째로 skip한다.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.near_duplicate import simhash
from argos.brain.simhash_storage import to_storage
from argos.config import settings
from argos.crawler import body_backfill
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
LONG = "Real article sentence. " * 40


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping ARG-309 refetch-bodies DB test")


@pytest.mark.asyncio
async def test_refetch_updates_raw_content_and_simhash_only():
    item_id = uuid.uuid4()
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with factory() as session:
            session.add(
                TechItem(
                    id=item_id,
                    title="ARG-309 fixture",
                    source_url=f"https://example-arg309.test/{item_id}",
                    raw_content="ARG-309 fixture\n\nshort",
                    summary="keep me",
                    category=CategoryType.MAINSTREAM,
                    trust_score=0.5,
                )
            )
            await session.commit()

        async def fetch(url):
            return {"title": "x", "raw_content": LONG, "source_url": url}

        async with factory() as session:
            rows = [r for r in await body_backfill.load_rows(session) if r.id == item_id]
            targets = body_backfill.select_targets(rows)
            assert len(targets) == 1
            summary = await body_backfill.refetch_and_apply(session, targets, fetch=fetch)
            assert summary.updated == 1

        async with factory() as session:
            row = (await session.execute(select(TechItem).where(TechItem.id == item_id))).scalar_one()
            expected = "ARG-309 fixture\n\n" + LONG.strip()
            assert row.raw_content == expected
            assert row.simhash == to_storage(simhash(expected))
            assert row.summary == "keep me"
    finally:
        async with factory() as session:
            await session.execute(delete(TechItem).where(TechItem.id == item_id))
            await session.commit()
        await engine.dispose()
