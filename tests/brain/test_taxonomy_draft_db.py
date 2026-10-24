"""fetch_candidates의 DB 통합 테스트 (ARG-311 T3) — 읽기 전용 쿼리가 실제로 도는지.

Postgres가 없으면 통째로 skip(릴리즈 CI 대비).
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.entity_extraction import ExtractedName
from argos.brain.entity_store import attach_names
from argos.brain.taxonomy_draft import fetch_candidates
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-311-draft-test.example.com/"
_KEY = "arg-311-test-zzgitzz"


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping ARG-311 draft DB test")


@pytest.fixture
async def session_factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with factory() as session:
            ids = (
                await session.execute(
                    TechItem.__table__.select()
                    .with_only_columns(TechItem.id)
                    .where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
                )
            ).all()
            if ids:
                await session.execute(
                    delete(DocumentEntity).where(
                        DocumentEntity.tech_item_id.in_([r[0] for r in ids])
                    )
                )
            await session.execute(
                delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
            )
            await session.execute(
                delete(Entity).where(Entity.normalized_key.like("arg-311-test-%"))
            )
            await session.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_fetch_candidates_returns_name_with_sentences(session_factory):
    async with session_factory() as session:
        for i in range(2):
            item = TechItem(
                id=uuid.uuid4(),
                title=f"ARG-311 draft test {i}",
                source_url=f"{_URL_PREFIX}{i}",
                raw_content="Intro text. ZZGitZZ tracks code history for teams. Other.",
                summary="Short summary.",
                category=CategoryType.ALPHA,
            )
            session.add(item)
            await session.flush()
            await attach_names(
                session, item.id, [ExtractedName(canonical=_KEY, surface="ZZGitZZ")]
            )
        await session.commit()

    async with session_factory() as session:
        found = await fetch_candidates(session, 500)

    mine = [c for c in found if c.name == "ZZGitZZ"]
    assert len(mine) == 1
    assert mine[0].sentences == ("ZZGitZZ tracks code history for teams.",)
