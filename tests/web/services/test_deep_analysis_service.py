"""심층 분석 서비스(ARG-336): 뷰 조립(DB 없음) + claim/save(DB)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from argos.brain.event_deep_analysis import DeepAnalysis, DeepAnalysisSection
from argos.config import settings
from argos.models.event_deep_analysis import DeepAnalysisStatus, EventDeepAnalysis
from argos.web.services.deep_analysis import _view_from_row, stale_notice
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


# --- 실패 이유 (ARG-338) ---------------------------------------------------


@pytest.mark.parametrize(
    "reason", ["NoEvidence", "ModelUnavailable", "Timeout", "BadFormat", "TooLarge"]
)
def test_failure_code_maps_generator_reasons(reason) -> None:
    from argos.brain.event_deep_analysis import DeepAnalysisError, DeepAnalysisFailure
    from argos.web.services.deep_analysis import failure_code

    assert failure_code(DeepAnalysisError(DeepAnalysisFailure(reason), "x")) == reason


@pytest.mark.parametrize("detail", ["HTTP 404", "model Not Found"])
def test_failure_code_model_missing_from_404_detail(detail) -> None:
    from argos.brain.event_deep_analysis import DeepAnalysisError, DeepAnalysisFailure
    from argos.web.services.deep_analysis import failure_code

    exc = DeepAnalysisError(DeepAnalysisFailure.ModelUnavailable, detail)
    assert failure_code(exc) == "ModelMissing"


def test_failure_code_unknown_for_other_exceptions() -> None:
    from argos.web.services.deep_analysis import failure_code

    assert failure_code(RuntimeError("boom")) == "Unknown"


def test_failure_message_unknown_code_falls_back() -> None:
    from argos.web.services.deep_analysis import FAILURE_TEXT, failure_message

    assert failure_message("Whatever") == FAILURE_TEXT["Unknown"]
    assert failure_message("Timeout") == "시간 안에 분석을 마치지 못했어요"
    assert failure_message(None) is None


def test_failed_view_carries_message() -> None:
    row = EventDeepAnalysis(event_id=uuid.uuid4())
    row.status = DeepAnalysisStatus.FAILED
    row.failure_reason = "Interrupted"
    view = _view_from_row(row, row.event_id, [uuid.uuid4()])
    assert view.failure_message == "웹이 다시 시작돼 분석이 중단됐어요"
    assert view.can_start is True


@pytestmark_db
@pytest.mark.asyncio
async def test_mark_failed_keeps_previous_result() -> None:
    from argos.web.services.deep_analysis import claim_deep_analysis, mark_failed, save_result

    engine, Session = _sessionmaker()
    event_id = await _seed_event(Session)
    try:
        ids = [uuid.uuid4(), uuid.uuid4()]
        async with Session() as session:
            await claim_deep_analysis(session, event_id)
        async with Session() as session:
            await save_result(session, event_id, _analysis(ids))
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is True
        async with Session() as session:
            await mark_failed(session, event_id, "Timeout")
        (row,) = await _row(Session, event_id)
        assert row.status == DeepAnalysisStatus.FAILED
        assert row.failure_reason == "Timeout"
        assert len(row.sections) == 5
        assert row.analyzed_at is not None
        assert row.doc_count == 2
    finally:
        await _cleanup(Session, event_id)
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_success_after_failure_clears_failure() -> None:
    from argos.web.services.deep_analysis import claim_deep_analysis, mark_failed, save_result

    engine, Session = _sessionmaker()
    event_id = await _seed_event(Session)
    try:
        async with Session() as session:
            await claim_deep_analysis(session, event_id)
        async with Session() as session:
            await mark_failed(session, event_id, "BadFormat")
        async with Session() as session:
            assert await claim_deep_analysis(session, event_id) is True
        async with Session() as session:
            await save_result(session, event_id, _analysis([uuid.uuid4()]))
        (row,) = await _row(Session, event_id)
        assert row.status is None
        assert row.failure_reason is None
    finally:
        await _cleanup(Session, event_id)
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fail_interrupted_runs_marks_waiting_and_running() -> None:
    from argos.web.services.deep_analysis import (
        claim_deep_analysis,
        fail_interrupted_runs,
        mark_failed,
        save_result,
    )

    engine, Session = _sessionmaker()
    ids = [await _seed_event(Session) for _ in range(4)]
    waiting, running, failed, done = ids
    try:
        async with Session() as s:
            await claim_deep_analysis(s, waiting, status=DeepAnalysisStatus.WAITING)
        async with Session() as s:
            await claim_deep_analysis(s, running)
        async with Session() as s:
            await claim_deep_analysis(s, failed)
        async with Session() as s:
            await mark_failed(s, failed, "Timeout")
        async with Session() as s:
            await claim_deep_analysis(s, done)
        async with Session() as s:
            await save_result(s, done, _analysis([uuid.uuid4()]))

        async with Session() as s:
            changed = await fail_interrupted_runs(s)
        assert changed >= 2  # 같은 DB의 다른 행도 정리될 수 있다

        for eid in (waiting, running):
            (row,) = await _row(Session, eid)
            assert row.status == DeepAnalysisStatus.FAILED
            assert row.failure_reason == "Interrupted"
        (row,) = await _row(Session, failed)
        assert row.failure_reason == "Timeout"
        (row,) = await _row(Session, done)
        assert row.status is None and len(row.sections) == 5
    finally:
        for eid in ids:
            await _cleanup(Session, eid)
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_mark_waiting_only_before_started() -> None:
    from argos.web.services.deep_analysis import (
        claim_deep_analysis,
        mark_failed,
        mark_running,
        mark_waiting,
    )

    engine, Session = _sessionmaker()
    claimed, started, failed = [await _seed_event(Session) for _ in range(3)]
    try:
        # 확인 때는 비어 있어 Running으로 잡았지만 아직 시작 전 → 대기로 바뀐다.
        async with Session() as s:
            await claim_deep_analysis(s, claimed)
        async with Session() as s:
            await mark_waiting(s, claimed)
        (row,) = await _row(Session, claimed)
        assert row.status == DeepAnalysisStatus.WAITING and row.started_at is None

        # 시작한 뒤(started_at 있음)에는 Running을 건드리지 않는다.
        async with Session() as s:
            await claim_deep_analysis(s, started)
        async with Session() as s:
            await mark_running(s, started)
        async with Session() as s:
            await mark_waiting(s, started)
        (row,) = await _row(Session, started)
        assert row.status == DeepAnalysisStatus.RUNNING and row.started_at is not None

        # Failed 행·없는 행은 건드리지 않는다.
        async with Session() as s:
            await claim_deep_analysis(s, failed)
        async with Session() as s:
            await mark_failed(s, failed, "Timeout")
        async with Session() as s:
            await mark_waiting(s, failed)
        (row,) = await _row(Session, failed)
        assert row.status == DeepAnalysisStatus.FAILED
    finally:
        for eid in (claimed, started, failed):
            await _cleanup(Session, eid)
        await engine.dispose()


def _ids(n: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(n)]


def test_stale_notice_none_when_same_set() -> None:
    a, b = _ids(2)
    assert stale_notice([a, b], [b, a]) is None


def test_stale_notice_added() -> None:
    a, b, c = _ids(3)
    assert stale_notice([a, b], [a, b, c]) == "새 기사 1건이 더 붙었어요 · 다시 분석"


def test_stale_notice_removed_only() -> None:
    a, b = _ids(2)
    assert stale_notice([a, b], [a]) == "기사 묶음이 달라졌어요 · 다시 분석"


def test_stale_notice_added_and_removed_prefers_added() -> None:
    a, b, c, d = _ids(4)
    assert stale_notice([a, b], [a, c, d]) == "새 기사 2건이 더 붙었어요 · 다시 분석"


def test_stale_notice_none_without_result() -> None:
    assert stale_notice(None, _ids(2)) is None


def test_view_sets_stale_notice_only_when_result_exists_and_not_in_progress() -> None:
    a, b, c = _ids(3)
    row = EventDeepAnalysis(
        event_id=uuid.uuid4(),
        sections=[{"heading": "h", "body": "b"}],
        doc_ids=[a, b],
        doc_count=2,
        status=None,
    )
    view = _view_from_row(row, row.event_id, [a, b, c])
    assert view.stale_notice == "새 기사 1건이 더 붙었어요 · 다시 분석"
    row.status = DeepAnalysisStatus.RUNNING
    assert _view_from_row(row, row.event_id, [a, b, c]).stale_notice is None
    row.status = DeepAnalysisStatus.WAITING
    assert _view_from_row(row, row.event_id, [a, b, c]).stale_notice is None
    assert _view_from_row(None, uuid.uuid4(), [a]).stale_notice is None
