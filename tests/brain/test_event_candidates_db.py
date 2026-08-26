"""event_candidates — 시간 창 안 코사인 상위 K 이웃 조회의 DB 통합 테스트 (ARG-265).

패턴은 ``tests/test_tech_event_tombstone_db.py``, ``tests/brain/test_entity_store_db.py``
와 같다: 모듈 스코프 ``session_factory`` 픽스처(NullPool) + 이 모듈이 만든 행만
정리 + Postgres가 없으면 통째로 skip.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.entity_extraction import ExtractedName
from argos.brain.entity_store import attach_names
from argos.brain.event_candidates import fetch_candidates, keywords_of
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-265-event-candidates-test.example.com/"


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-265 event_candidates DB "
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
            item_ids_result = await session.execute(
                TechItem.__table__.select().with_only_columns(TechItem.id).where(
                    TechItem.source_url.like(f"{_URL_PREFIX}%")
                )
            )
            item_ids = [row[0] for row in item_ids_result.all()]
            if item_ids:
                await session.execute(
                    delete(EventDocument).where(EventDocument.tech_item_id.in_(item_ids))
                )
                await session.execute(
                    delete(DocumentEntity).where(DocumentEntity.tech_item_id.in_(item_ids))
                )
            await session.execute(
                delete(TechEvent).where(TechEvent.title.like("ARG-265 event candidates test%"))
            )
            await session.execute(
                delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
            )
            await session.execute(
                delete(Entity).where(Entity.normalized_key.like("arg-265-test-%"))
            )
            await session.commit()
        await engine.dispose()


def _embedding(seed: float) -> list[float]:
    """768차원 임베딩. 첫 성분만 seed로 바꿔 서로 다른(혹은 같은) 코사인을 만든다."""
    vec = [0.0] * 768
    vec[0] = seed
    vec[1] = 1.0
    return vec


async def _make_item(
    session,
    suffix: str,
    *,
    embedding: list[float] | None,
    published_at: datetime | None,
    created_at: datetime | None = None,
    summary: str = "",
) -> uuid.UUID:
    item = TechItem(
        id=uuid.uuid4(),
        title=f"ARG-265 event candidates test {suffix}",
        source_url=f"{_URL_PREFIX}{suffix}",
        raw_content="x",
        summary=summary or None,
        category=CategoryType.ALPHA,
        embedding=embedding,
        published_at=published_at,
    )
    session.add(item)
    await session.flush()
    if created_at is not None:
        # created_at은 TimestampMixin의 server_default라 직접 값을 지정하려면
        # flush 후 UPDATE가 필요하다.
        await session.execute(
            TechItem.__table__.update()
            .where(TechItem.id == item.id)
            .values(created_at=created_at)
        )
    return item.id


# 실제 "지금"을 쓴다 — created_at은 DB server_default(func.now())라서 고정된
# 과거/미래 상수를 :at로 주면 published_at=None 문서의 created_at(실제 현재
# 시각)이 그 상수보다 뒤라 창 밖으로 밀려나 버린다.
NOW = datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_only_documents_inside_the_window_come_back(session_factory):
    async with session_factory() as session:
        inside_id = await _make_item(
            session,
            "inside",
            embedding=_embedding(1.0),
            published_at=NOW - timedelta(days=1),
        )
        outside_id = await _make_item(
            session,
            "outside",
            embedding=_embedding(1.0),
            published_at=NOW - timedelta(days=15),
        )
        await session.commit()

        results = await fetch_candidates(session, embedding=_embedding(1.0), at=NOW, window_days=14)
        ids = {c.tech_item_id for c in results}
        assert inside_id in ids
        assert outside_id not in ids


@pytest.mark.asyncio
async def test_a_null_published_at_falls_back_to_created_at(session_factory):
    async with session_factory() as session:
        item_id = await _make_item(
            session,
            "null-published",
            embedding=_embedding(1.0),
            published_at=None,
        )
        await session.commit()

        # created_at은 서버 default(func.now())라 방금 커밋된 실제 시각이다.
        # 그 시각보다 확실히 뒤인 "지금"으로 조회해야 창 안에 잡힌다.
        at = datetime.now(timezone.utc)
        results = await fetch_candidates(session, embedding=_embedding(1.0), at=at, window_days=14)
        ids = {c.tech_item_id for c in results}
        assert item_id in ids


@pytest.mark.asyncio
async def test_documents_without_an_embedding_are_skipped(session_factory):
    async with session_factory() as session:
        no_embedding_id = await _make_item(
            session,
            "no-embedding",
            embedding=None,
            published_at=NOW - timedelta(days=1),
        )
        await session.commit()

        results = await fetch_candidates(session, embedding=_embedding(1.0), at=NOW, window_days=14)
        ids = {c.tech_item_id for c in results}
        assert no_embedding_id not in ids


@pytest.mark.asyncio
async def test_results_are_ordered_by_cosine_then_id(session_factory):
    async with session_factory() as session:
        id_a = await _make_item(
            session, "tie-a", embedding=_embedding(1.0), published_at=NOW - timedelta(days=1)
        )
        id_b = await _make_item(
            session, "tie-b", embedding=_embedding(1.0), published_at=NOW - timedelta(days=1)
        )
        await session.commit()

        first = await fetch_candidates(session, embedding=_embedding(1.0), at=NOW, window_days=14)
        second = await fetch_candidates(session, embedding=_embedding(1.0), at=NOW, window_days=14)

        tie_ids_first = [c.tech_item_id for c in first if c.tech_item_id in (id_a, id_b)]
        tie_ids_second = [c.tech_item_id for c in second if c.tech_item_id in (id_a, id_b)]
        assert tie_ids_first == tie_ids_second == sorted([id_a, id_b])


@pytest.mark.asyncio
async def test_the_limit_caps_the_result(session_factory):
    async with session_factory() as session:
        for suffix in ("limit-a", "limit-b", "limit-c"):
            await _make_item(
                session,
                suffix,
                embedding=_embedding(1.0),
                published_at=NOW - timedelta(days=1),
            )
        await session.commit()

        results = await fetch_candidates(
            session, embedding=_embedding(1.0), at=NOW, window_days=14, limit=2
        )
        assert len(results) == 2


@pytest.mark.asyncio
async def test_neighbours_carry_their_event_links_and_names(session_factory):
    async with session_factory() as session:
        item_id = await _make_item(
            session,
            "with-links",
            embedding=_embedding(1.0),
            published_at=NOW - timedelta(days=1),
            summary="Anthropic ships Claude",
        )
        event = TechEvent(
            title="ARG-265 event candidates test — event",
            occurred_at=NOW - timedelta(days=1),
        )
        session.add(event)
        await session.flush()
        session.add(EventDocument(event_id=event.id, tech_item_id=item_id))
        await attach_names(
            session, item_id, [ExtractedName(canonical="arg-265-test-anthropic", surface="Anthropic")]
        )
        await session.commit()

        results = await fetch_candidates(session, embedding=_embedding(1.0), at=NOW, window_days=14)
        match = next(c for c in results if c.tech_item_id == item_id)
        assert match.event_ids == (event.id,)
        assert match.features.names == frozenset({"arg-265-test-anthropic"})


@pytest.mark.asyncio
async def test_an_empty_window_returns_an_empty_list(session_factory):
    async with session_factory() as session:
        results = await fetch_candidates(
            session,
            embedding=_embedding(1.0),
            at=NOW - timedelta(days=9999),
            window_days=1,
        )
        assert results == []


@pytest.mark.asyncio
async def test_the_document_itself_is_excluded(session_factory):
    async with session_factory() as session:
        item_id = await _make_item(
            session,
            "exclude-self",
            embedding=_embedding(1.0),
            published_at=NOW - timedelta(days=1),
        )
        await session.commit()

        results = await fetch_candidates(
            session, embedding=_embedding(1.0), at=NOW, window_days=14, exclude_id=item_id
        )
        ids = {c.tech_item_id for c in results}
        assert item_id not in ids


def test_keywords_of_lowercases_and_splits_words():
    assert keywords_of("Claude Sonnet 5!") == frozenset({"claude", "sonnet", "5"})
    assert keywords_of(None) == frozenset()
    assert keywords_of("") == frozenset()
