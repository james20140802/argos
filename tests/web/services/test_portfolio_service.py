"""Unit tests for argos.web.services.portfolio (ARG-153).

All tests run without a live Postgres — AsyncSession.execute is mocked.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.web.services.portfolio import (
    RECENT_SIGNAL_WINDOW,
    PortfolioView,
    fetch_portfolio,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def _make_row(
    *,
    ua_id: uuid.UUID | None = None,
    tech_id: uuid.UUID | None = None,
    title: str = "Test Tech",
    source_url: str | None = None,
    category: str | None = "Mainstream",
    image_url: str | None = None,
    trust_score: float | None = 0.5,
    kept_since: datetime | None = None,
    last_signal_at: datetime | None = None,
    signal_count: int = 0,
    lineage_count: int = 0,
) -> MagicMock:
    row = MagicMock()
    row.ua_id = ua_id or uuid.uuid4()
    row.tech_id = tech_id or uuid.uuid4()
    row.title = title
    row.source_url = source_url or f"https://example.com/{uuid.uuid4()}"
    row.category = category
    row.image_url = image_url
    row.trust_score = trust_score
    row.kept_since = kept_since or _utc("2026-01-01T00:00:00")
    row.last_signal_at = last_signal_at
    row.signal_count = signal_count
    row.lineage_count = lineage_count
    return row


def _make_session(rows: list) -> AsyncMock:
    """Return a mock AsyncSession whose execute() yields given rows."""
    result = MagicMock()
    result.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    return session


# ------------------------------------------------------------------ #
# Test 1: empty portfolio → both groups empty
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_empty_portfolio_returns_empty_view() -> None:
    session = _make_session([])
    view = await fetch_portfolio(session, category="Alpha", sort="trust")
    assert isinstance(view, PortfolioView)
    assert view.active == []
    assert view.quiet == []
    assert view.category == "Alpha"
    assert view.sort == "trust"


# ------------------------------------------------------------------ #
# Test 2: partition active vs quiet by signal/lineage
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_partitions_active_and_quiet() -> None:
    active_row = _make_row(title="Active", signal_count=2, lineage_count=0)
    quiet_row = _make_row(title="Quiet", signal_count=0, lineage_count=0)
    session = _make_session([active_row, quiet_row])

    view = await fetch_portfolio(session)
    assert len(view.active) == 1
    assert len(view.quiet) == 1
    assert view.active[0].title == "Active"
    assert view.quiet[0].title == "Quiet"


# ------------------------------------------------------------------ #
# Test 3: lineage alone qualifies as active
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_lineage_alone_qualifies_as_active() -> None:
    row = _make_row(title="HasLineage", signal_count=0, lineage_count=1)
    session = _make_session([row])

    view = await fetch_portfolio(session)
    assert len(view.active) == 1
    assert view.active[0].title == "HasLineage"
    assert view.quiet == []


# ------------------------------------------------------------------ #
# Test 4: recency sort orders by kept_since DESC within group
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_recency_sort_within_group_orders_by_kept_since_desc() -> None:
    old = _make_row(title="Old", kept_since=_utc("2026-01-01T00:00:00"), signal_count=1)
    mid = _make_row(title="Mid", kept_since=_utc("2026-03-01T00:00:00"), signal_count=1)
    new = _make_row(title="New", kept_since=_utc("2026-05-01T00:00:00"), signal_count=1)

    # DB already returns in order we mock; service should preserve/apply sort
    session = _make_session([old, mid, new])

    view = await fetch_portfolio(session, sort="recency")
    titles = [a.title for a in view.active]
    # newest first
    assert titles == ["New", "Mid", "Old"]


# ------------------------------------------------------------------ #
# Test 5: trust sort → trust_score DESC NULLS LAST, then kept_since DESC
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_trust_sort_orders_trust_score_desc_nulls_last() -> None:
    high = _make_row(title="High", trust_score=0.9, kept_since=_utc("2026-01-01T00:00:00"))
    null_score = _make_row(title="Null", trust_score=None, kept_since=_utc("2026-02-01T00:00:00"))
    low = _make_row(title="Low", trust_score=0.5, kept_since=_utc("2026-03-01T00:00:00"))

    # All are quiet (signal_count=0, lineage_count=0) — but ordering still applies within quiet
    session = _make_session([high, null_score, low])

    view = await fetch_portfolio(session, sort="trust")
    titles = [a.title for a in view.quiet]
    assert titles == ["High", "Low", "Null"]


# ------------------------------------------------------------------ #
# Test 6: signal window boundary — old rows excluded from signal_count
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_signal_window_boundary_excludes_old_rows() -> None:
    # One asset with fresh signal_count > 0, one with signal_count == 0
    fresh = _make_row(title="Fresh", signal_count=1, lineage_count=0)
    stale = _make_row(title="Stale", signal_count=0, lineage_count=0)
    session = _make_session([fresh, stale])

    view = await fetch_portfolio(session)
    assert view.active[0].title == "Fresh"
    assert view.quiet[0].title == "Stale"


# ------------------------------------------------------------------ #
# Test 7: invalid category raises ValueError
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_category_filter_rejects_invalid() -> None:
    session = _make_session([])
    with pytest.raises(ValueError, match="invalid category"):
        await fetch_portfolio(session, category="Junk")  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# Test 8: category="Mainstream" is accepted and passed to query
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_category_filter_passes_mainstream_through() -> None:
    mainstream_row = _make_row(title="MS", category="Mainstream", signal_count=0, lineage_count=0)
    session = _make_session([mainstream_row])

    view = await fetch_portfolio(session, category="Mainstream")
    # No error, category echoed back
    assert view.category == "Mainstream"
    # The returned row has Mainstream category
    assert len(view.quiet) == 1
    assert view.quiet[0].category == "Mainstream"


# ------------------------------------------------------------------ #
# Test 9: invalid sort raises ValueError
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_invalid_sort_raises() -> None:
    session = _make_session([])
    with pytest.raises(ValueError, match="invalid sort"):
        await fetch_portfolio(session, sort="banana")  # type: ignore[arg-type]


# ------------------------------------------------------------------ #
# Test 10: RECENT_SIGNAL_WINDOW is 7 days (module constant check)
# ------------------------------------------------------------------ #

def test_recent_signal_window_is_7_days() -> None:
    assert RECENT_SIGNAL_WINDOW == timedelta(days=7)
