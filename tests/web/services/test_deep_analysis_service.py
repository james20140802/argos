"""심층 분석 서비스(ARG-336): 뷰 조립(DB 없음) + claim/save(DB)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from argos.brain.event_deep_analysis import DeepAnalysis, DeepAnalysisSection
from argos.config import settings
from argos.models.event_deep_analysis import DeepAnalysisStatus, EventDeepAnalysis
from argos.web.services.deep_analysis import _view_from_row
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url

pytestmark_db = pytest.mark.skipif(
    not _db_reachable(_DB_URL),
    reason="pgvector DB not reachable — skipping ARG-336 DB-backed tests",
)

HEADINGS = ("핵심 기술·접근", "성숙도", "생태계와의 관계", "위험·한계", "추적할 만한가")


def test_view_without_row_can_start_when_event_has_docs() -> None:
    eid = uuid.uuid4()
    view = _view_from_row(None, eid, [uuid.uuid4(), uuid.uuid4()])
    assert view.can_start is True
    assert view.sections == ()
    assert view.status is None
    assert view.event_id == eid


def test_view_cannot_start_without_docs() -> None:
    view = _view_from_row(None, uuid.uuid4(), [])
    assert view.can_start is False


def test_view_in_progress_blocks_start() -> None:
    row = EventDeepAnalysis(event_id=uuid.uuid4(), status=DeepAnalysisStatus.RUNNING)
    view = _view_from_row(row, row.event_id, [uuid.uuid4()])
    assert view.in_progress is True
    assert view.can_start is False


def test_view_maps_saved_sections() -> None:
    row = EventDeepAnalysis(
        event_id=uuid.uuid4(),
        sections=[{"heading": h, "body": f"{h} 본문"} for h in HEADINGS],
        analyzed_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        doc_count=2,
        status=None,
    )
    view = _view_from_row(row, row.event_id, [uuid.uuid4()])
    assert [s.heading for s in view.sections] == list(HEADINGS)
    assert view.doc_count == 2
    assert view.in_progress is False
    assert view.can_start is True


def _analysis(doc_ids: list[uuid.UUID]) -> DeepAnalysis:
    return DeepAnalysis(
        sections=tuple(DeepAnalysisSection(h, f"{h} 본문") for h in HEADINGS),
        doc_ids=tuple(doc_ids),
        doc_count=len(doc_ids),
        model="test-model",
    )


async def _seed_event(Session) -> uuid.UUID:
    from argos.models.tech_event import TechEvent

    async with Session() as session:
        ev = TechEvent(
            title="arg336 사건",
            summary="s",
            event_type="Launch",
            occurred_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
        session.add(ev)
        await session.commit()
        return ev.id


async def _cleanup(Session, event_id: uuid.UUID) -> None:
    from sqlalchemy import delete

    from argos.models.tech_event import TechEvent

    async with Session() as session:
        await session.execute(delete(TechEvent).where(TechEvent.id == event_id))
        await session.commit()


def _sessionmaker():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    return engine, async_sessionmaker(bind=engine, expire_on_commit=False)


async def _row(Session, event_id: uuid.UUID) -> list[EventDeepAnalysis]:
    from sqlalchemy import select

    async with Session() as session:
        return list(
            (
                await session.execute(
                    select(EventDeepAnalysis).where(EventDeepAnalysis.event_id == event_id)
                )
            ).scalars()
        )


@pytestmark_db
@pytest.mark.asyncio
async def test_claim_twice_second_returns_false() -> None:
    from argos.web.services.deep_analysis import claim_deep_analysis

    engine, Session = _sessionmaker()
    event_id = await _seed_event(Session)
    try:
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is True
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is False
        rows = await _row(Session, event_id)
        assert len(rows) == 1
        assert rows[0].status == DeepAnalysisStatus.RUNNING
    finally:
        await _cleanup(Session, event_id)
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_claim_after_success_keeps_previous_result() -> None:
    from argos.web.services.deep_analysis import claim_deep_analysis, save_result

    engine, Session = _sessionmaker()
    event_id = await _seed_event(Session)
    try:
        ids = [uuid.uuid4(), uuid.uuid4()]
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is True
        async with Session() as session:
            await save_result(session, event_id, _analysis(ids))
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is True
        (row,) = await _row(Session, event_id)
        assert row.status == DeepAnalysisStatus.RUNNING
        assert row.analyzed_at is not None
        assert row.doc_count == 2
        assert len(row.sections) == 5
    finally:
        await _cleanup(Session, event_id)
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_save_result_clears_status_and_stores_doc_ids() -> None:
    from argos.web.services.deep_analysis import claim_deep_analysis, save_result

    engine, Session = _sessionmaker()
    event_id = await _seed_event(Session)
    try:
        ids = [uuid.uuid4(), uuid.uuid4()]
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is True
        async with Session() as session:
            await save_result(session, event_id, _analysis(ids))
        (row,) = await _row(Session, event_id)
        assert row.status is None
        assert list(row.doc_ids) == ids
        assert row.doc_count == 2
    finally:
        await _cleanup(Session, event_id)
        await engine.dispose()
