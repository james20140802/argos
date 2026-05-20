from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_dataclasses_exposed():
    from argos.brain.weekly_report import (
        WeeklyKeepItem,
        WeeklyKeepReport,
        build_weekly_keep_report,
    )

    item = WeeklyKeepItem(
        tech_id=uuid.uuid4(),
        title="t",
        signals_7d=0,
        successions_7d=0,
        last_monitored_at=None,
    )
    rep = WeeklyKeepReport(
        total_keep_count=0,
        items=[],
        window_start=datetime(2026, 5, 13, 12, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 20, 12, tzinfo=timezone.utc),
    )
    assert rep.total_keep_count == 0
    assert item.signals_7d == 0
    assert callable(build_weekly_keep_report)


@pytest.mark.asyncio
async def test_empty_keep_returns_zero_total():
    from argos.brain.weekly_report import build_weekly_keep_report

    empty_result = MagicMock()
    empty_result.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=empty_result)

    now_utc = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
    report = await build_weekly_keep_report(session, now_utc=now_utc)

    assert report.total_keep_count == 0
    assert report.items == []
    assert report.window_end == now_utc
    assert report.window_start == now_utc - timedelta(days=7)


def _result(rows):
    r = MagicMock()
    r.all.return_value = rows
    return r


@pytest.mark.asyncio
async def test_single_keep_no_activity():
    from argos.brain.weekly_report import build_weekly_keep_report

    asset_id = uuid.uuid4()
    tech_id = uuid.uuid4()
    monitored = datetime(2026, 5, 18, 10, tzinfo=timezone.utc)
    keep_rows = [(asset_id, tech_id, monitored, "Tech A")]

    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[_result(keep_rows), _result([]), _result([])]
    )

    now_utc = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    report = await build_weekly_keep_report(session, now_utc=now_utc)

    assert report.total_keep_count == 1
    assert len(report.items) == 1
    item = report.items[0]
    assert item.tech_id == tech_id
    assert item.title == "Tech A"
    assert item.signals_7d == 0
    assert item.successions_7d == 0
    assert item.last_monitored_at == monitored


@pytest.mark.asyncio
async def test_signals_and_successions_counted():
    from argos.brain.weekly_report import build_weekly_keep_report

    asset_a, asset_b, asset_c = (uuid.uuid4() for _ in range(3))
    tech_a, tech_b, tech_c = (uuid.uuid4() for _ in range(3))
    keep_rows = [
        (asset_a, tech_a, None, "A"),
        (asset_b, tech_b, None, "B"),
        (asset_c, tech_c, None, "C"),
    ]
    signals = [(asset_a, 3), (asset_b, 1)]  # asset_c omitted -> 0
    successions = [(tech_a, 2)]  # tech_b / tech_c omitted -> 0

    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[_result(keep_rows), _result(signals), _result(successions)]
    )

    now_utc = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    report = await build_weekly_keep_report(session, now_utc=now_utc)

    by_tech = {item.tech_id: item for item in report.items}
    assert by_tech[tech_a].signals_7d == 3
    assert by_tech[tech_a].successions_7d == 2
    assert by_tech[tech_b].signals_7d == 1
    assert by_tech[tech_b].successions_7d == 0
    assert by_tech[tech_c].signals_7d == 0
    assert by_tech[tech_c].successions_7d == 0


@pytest.mark.asyncio
async def test_items_ordered_by_title():
    from argos.brain.weekly_report import build_weekly_keep_report

    # Q1 is expected to ORDER BY title ASC; the mock honors that order.
    keep_rows = [
        (uuid.uuid4(), uuid.uuid4(), None, "Alpha"),
        (uuid.uuid4(), uuid.uuid4(), None, "Beta"),
        (uuid.uuid4(), uuid.uuid4(), None, "Gamma"),
    ]
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[_result(keep_rows), _result([]), _result([])]
    )

    report = await build_weekly_keep_report(
        session, now_utc=datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    )

    titles = [item.title for item in report.items]
    assert titles == sorted(titles)


@pytest.mark.asyncio
async def test_keep_status_filter_in_query():
    from argos.brain.weekly_report import build_weekly_keep_report

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_result([]), _result([]), _result([])])
    await build_weekly_keep_report(
        session, now_utc=datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    )

    # First execute call = Q1; its compiled SQL must reference the Keep filter.
    first_stmt = session.execute.await_args_list[0].args[0]
    compiled = str(first_stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "status" in compiled
    assert "Keep" in compiled


@pytest.mark.asyncio
async def test_window_bounds_passed_to_signal_query():
    from argos.brain.weekly_report import build_weekly_keep_report

    keep_rows = [(uuid.uuid4(), uuid.uuid4(), None, "X")]
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[_result(keep_rows), _result([]), _result([])]
    )
    now_utc = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    await build_weekly_keep_report(session, now_utc=now_utc)

    # Q2 is the second execute call (signals).
    stmt = session.execute.await_args_list[1].args[0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "track_history" in compiled
    assert "changed_at" in compiled
    # Window literals appear in the compiled SQL.
    assert "2026-05-13 12:00:00" in compiled
    assert "2026-05-20 12:00:00" in compiled


@pytest.mark.asyncio
async def test_default_now_uses_utc():
    from argos.brain.weekly_report import build_weekly_keep_report

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_result([])])  # short-circuit
    before = datetime.now(timezone.utc)
    report = await build_weekly_keep_report(session)
    after = datetime.now(timezone.utc)

    assert report.window_end.tzinfo == timezone.utc
    assert before <= report.window_end <= after
    assert report.window_end - report.window_start == timedelta(days=7)
