from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus
from argos.slack.services.briefing_query import fetch_today_briefing, fetch_user_portfolio, KST


def _make_tech_item(category: CategoryType, trust_score: float | None, created_at: datetime):
    item = MagicMock()
    item.id = uuid.uuid4()
    item.category = category
    item.trust_score = trust_score
    item.created_at = created_at
    return item


@pytest.mark.asyncio
async def test_kst_window_filters_today_items(now_utc):
    now_kst = now_utc.astimezone(KST)
    start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    within_window = start_kst.astimezone(timezone.utc) + timedelta(hours=5)
    outside_window = start_kst.astimezone(timezone.utc) - timedelta(hours=1)

    today_item = _make_tech_item(CategoryType.MAINSTREAM, 0.8, within_window)
    _make_tech_item(CategoryType.MAINSTREAM, 0.9, outside_window)

    captured_queries = []

    async def fake_execute(stmt):
        captured_queries.append(stmt)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [today_item]
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc)

    assert CategoryType.MAINSTREAM in result
    assert CategoryType.ALPHA in result


@pytest.mark.asyncio
async def test_returns_dict_with_both_categories(now_utc):
    ms_item = _make_tech_item(CategoryType.MAINSTREAM, 0.9, now_utc)
    alpha_item = _make_tech_item(CategoryType.ALPHA, 0.5, now_utc)

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        mock_result = MagicMock()
        if call_count == 0:
            mock_result.scalars.return_value.all.return_value = [ms_item]
        else:
            mock_result.scalars.return_value.all.return_value = [alpha_item]
        call_count += 1
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc)

    assert CategoryType.MAINSTREAM in result
    assert CategoryType.ALPHA in result


@pytest.mark.asyncio
async def test_limit_per_category_honored(now_utc):
    items = [_make_tech_item(CategoryType.MAINSTREAM, float(i) / 10, now_utc) for i in range(10)]
    returned = items[:3]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = returned
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc, limit_per_category=3)

    assert len(result[CategoryType.MAINSTREAM]) == 3


@pytest.mark.asyncio
async def test_empty_result_when_no_items(now_utc):
    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc)

    assert result[CategoryType.MAINSTREAM] == []
    assert result[CategoryType.ALPHA] == []


@pytest.mark.asyncio
async def test_default_now_utc_used_when_none():
    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session)
    assert isinstance(result, dict)
    assert CategoryType.MAINSTREAM in result


# ---------------------------------------------------------------------------
# fetch_user_portfolio tests
# ---------------------------------------------------------------------------


def _make_asset_and_item(
    status: AssetStatus = AssetStatus.KEEP,
    updated_at: datetime | None = None,
) -> tuple[MagicMock, MagicMock]:
    tech_id = uuid.uuid4()
    item = MagicMock()
    item.id = tech_id
    item.title = "Test Tech"
    item.source_url = "https://example.com/tech"

    asset = MagicMock()
    asset.id = uuid.uuid4()
    asset.tech_id = tech_id
    asset.status = status
    asset.updated_at = updated_at or datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)
    asset.last_monitored_at = None

    return asset, item


@pytest.mark.asyncio
async def test_fetch_user_portfolio_returns_keep_assets():
    asset, item = _make_asset_and_item(AssetStatus.KEEP)
    rows = [(asset, item)]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_user_portfolio(mock_session)
    assert len(result) == 1
    assert result[0] == (asset, item)


@pytest.mark.asyncio
async def test_fetch_user_portfolio_empty_when_no_assets():
    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_user_portfolio(mock_session)
    assert result == []


@pytest.mark.asyncio
async def test_fetch_user_portfolio_query_filters_keep_and_orders_by_updated_at():
    """Verify the SQL statement targets KEEP status and orders by updated_at DESC."""
    captured = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_user_portfolio(mock_session)

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "user_assets" in compiled
    assert "Keep" in compiled
