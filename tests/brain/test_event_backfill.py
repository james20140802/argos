import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain.event_backfill import BackfillDoc, plan_backfill
from argos.brain.event_scoring import DocumentFeatures
from argos.config import settings

AT = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _doc(embedding, names, at, title="t", summary="s") -> BackfillDoc:
    return BackfillDoc(
        tech_item_id=uuid.uuid4(),
        features=DocumentFeatures(
            embedding=tuple(embedding),
            names=frozenset(names),
            at=at,
            keywords=frozenset(summary.split()),
        ),
        title=title,
        summary=summary,
    )


def _session_without_db_neighbours() -> MagicMock:
    """``db_candidate_source``가 항상 빈 목록을 주는 세션 목.

    ``begin_nested``는 async context manager여야 한다 — 뒤 태스크의 실행·
    재명명 경로가 ``async with session.begin_nested():``를 쓴다.
    """

    @asynccontextmanager
    async def _nested():
        yield None

    session = MagicMock()
    session.begin_nested = MagicMock(side_effect=lambda: _nested())
    return session


@pytest.mark.asyncio
async def test_two_similar_documents_land_in_one_virtual_event(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    docs = [
        _doc([1.0, 0.0], ["anthropic"], AT, summary="claude 5 shipped"),
        _doc([1.0, 0.0], ["anthropic"], AT + timedelta(hours=2), summary="claude 5 shipped"),
    ]
    plan = await plan_backfill(
        _session_without_db_neighbours(), docs, config=settings.user.event_detection
    )
    assert plan.new_event_count == 1
    assert len({a.event_id for a in plan.assignments}) == 1
    assert [a.created for a in plan.assignments] == [True, False]


@pytest.mark.asyncio
async def test_unrelated_documents_create_separate_events(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    docs = [
        _doc([1.0, 0.0], ["anthropic"], AT, summary="claude shipped"),
        _doc([0.0, 1.0], ["mistral"], AT + timedelta(days=13), summary="totally other news"),
    ]
    plan = await plan_backfill(
        _session_without_db_neighbours(), docs, config=settings.user.event_detection
    )
    assert plan.new_event_count == 2
    assert plan.size_distribution == {1: 2}


@pytest.mark.asyncio
async def test_plan_never_writes_to_the_session(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    session = _session_without_db_neighbours()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    await plan_backfill(
        session,
        [_doc([1.0, 0.0], ["anthropic"], AT)],
        config=settings.user.event_detection,
    )
    session.add.assert_not_called()
    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_size_distribution_counts_events_by_document_count(monkeypatch):
    from argos.brain import event_backfill

    monkeypatch.setattr(
        event_backfill, "db_candidate_source", AsyncMock(return_value=[])
    )
    docs = [
        _doc([1.0, 0.0], ["anthropic"], AT, summary="claude shipped"),
        _doc([1.0, 0.0], ["anthropic"], AT + timedelta(hours=1), summary="claude shipped"),
        _doc([0.0, 1.0], ["mistral"], AT + timedelta(days=13), summary="other news here"),
    ]
    plan = await plan_backfill(
        _session_without_db_neighbours(), docs, config=settings.user.event_detection
    )
    assert plan.size_distribution == {2: 1, 1: 1}
