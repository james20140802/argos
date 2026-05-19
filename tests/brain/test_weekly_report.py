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
