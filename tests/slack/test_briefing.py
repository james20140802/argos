from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.models.tech_item import CategoryType
from argos.slack.briefing import dispatch_daily_briefing


def _make_item(title: str, trust_score: float | None = 0.5):
    return SimpleNamespace(
        id=uuid.uuid4(),
        title=title,
        source_url=f"https://example.com/{title}",
        trust_score=trust_score,
        summary=None,
    )


def _patch_dispatch(items_by_category, *, header_ts: str = "1700000000.001"):
    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": header_ts})

    mock_app = MagicMock()
    mock_app.client = mock_client

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    return mock_client, mock_app, session_ctx


@pytest.mark.asyncio
async def test_dispatch_posts_empty_state_when_all_categories_empty():
    """When no items found, dispatch must post an empty-state message to Slack."""
    items_by_category = {CategoryType.MAINSTREAM: [], CategoryType.ALPHA: []}
    mock_client, mock_app, session_ctx = _patch_dispatch(items_by_category)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        await dispatch_daily_briefing(channel="C999")

    # Must post exactly one message containing the empty-state text
    mock_client.chat_postMessage.assert_awaited_once()
    call_kwargs = mock_client.chat_postMessage.call_args.kwargs
    assert call_kwargs["channel"] == "C999"
    assert "오늘 브리핑할 최신 소식이 없습니다" in call_kwargs.get("text", "")


@pytest.mark.asyncio
async def test_dispatch_posts_header_then_threaded_items():
    items_by_category = {
        CategoryType.MAINSTREAM: [_make_item("Stream-A"), _make_item("Stream-B")],
        CategoryType.ALPHA: [_make_item("Alpha-A")],
    }
    mock_client, mock_app, session_ctx = _patch_dispatch(items_by_category)

    fetch_mock = AsyncMock(return_value=items_by_category)
    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        fetch_mock,
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        from argos.config import settings

        result = await dispatch_daily_briefing(channel="C999")

    assert result == "1700000000.001"
    # fetch_today_briefing should receive the configured limit_per_category
    assert fetch_mock.await_args.kwargs["limit_per_category"] == (
        settings.user.briefing.limit_per_category
    )
    calls = mock_client.chat_postMessage.await_args_list
    # 1 header + 2 category headers + 3 item replies
    assert len(calls) == 1 + 2 + 3

    header_call = calls[0]
    assert "thread_ts" not in header_call.kwargs
    assert header_call.kwargs["channel"] == "C999"

    for follow_up in calls[1:]:
        assert follow_up.kwargs["thread_ts"] == "1700000000.001"
        assert "reply_broadcast" not in follow_up.kwargs
        assert follow_up.kwargs["channel"] == "C999"


@pytest.mark.asyncio
async def test_dispatch_skips_empty_category_section():
    items_by_category = {
        CategoryType.MAINSTREAM: [_make_item("M-1")],
        CategoryType.ALPHA: [],
    }
    mock_client, mock_app, session_ctx = _patch_dispatch(items_by_category)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        await dispatch_daily_briefing(channel="C999")

    calls = mock_client.chat_postMessage.await_args_list
    # 1 header + 1 mainstream category header + 1 mainstream item; no alpha.
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_dispatch_stamps_briefed_at_after_posting():
    """dispatch_daily_briefing must commit briefed_at on every posted item."""
    items_by_category = {
        CategoryType.MAINSTREAM: [_make_item("M-1"), _make_item("M-2")],
        CategoryType.ALPHA: [_make_item("A-1")],
    }
    mock_client, mock_app, _ = _patch_dispatch(items_by_category)

    mock_session = AsyncMock()
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        await dispatch_daily_briefing(channel="C999")

    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_does_not_stamp_when_no_items_posted():
    """dispatch_daily_briefing must not issue a DB update when all categories are empty."""
    items_by_category = {CategoryType.MAINSTREAM: [], CategoryType.ALPHA: []}
    mock_client, mock_app, _ = _patch_dispatch(items_by_category)

    mock_session = AsyncMock()
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        await dispatch_daily_briefing(channel="C999")

    mock_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_item_messages_enable_unfurl():
    items_by_category = {
        CategoryType.MAINSTREAM: [_make_item("M-1")],
        CategoryType.ALPHA: [],
    }
    mock_client, mock_app, session_ctx = _patch_dispatch(items_by_category)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        await dispatch_daily_briefing(channel="C999")

    item_call = mock_client.chat_postMessage.await_args_list[-1]
    assert item_call.kwargs.get("unfurl_links") is True
    assert item_call.kwargs.get("unfurl_media") is True


# ---------------------------------------------------------------------------
# Weekly briefing dispatch (ARG-123)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_weekly_briefing_posts_single_message():
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from argos.brain.weekly_report import WeeklyKeepItem, WeeklyKeepReport
    from argos.slack.briefing import dispatch_weekly_briefing

    monitored = _dt(2026, 5, 18, 10, tzinfo=_tz.utc)
    item = WeeklyKeepItem(
        tech_id=uuid.uuid4(),
        title="Tech A",
        signals_7d=2,
        successions_7d=1,
        last_monitored_at=monitored,
    )
    now_utc = _dt(2026, 5, 20, 12, tzinfo=_tz.utc)
    report = WeeklyKeepReport(
        total_keep_count=1,
        items=[item],
        window_start=now_utc - _td(days=7),
        window_end=now_utc,
    )

    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1700000000.999"})
    mock_app = MagicMock()
    mock_app.client = mock_client

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.build_weekly_keep_report",
        AsyncMock(return_value=report),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        result = await dispatch_weekly_briefing(channel="C999")

    assert result == "1700000000.999"
    # Exactly ONE chat_postMessage call (single message, NOT threaded).
    assert len(mock_client.chat_postMessage.await_args_list) == 1
    call = mock_client.chat_postMessage.await_args_list[0]
    assert call.kwargs["channel"] == "C999"
    assert "thread_ts" not in call.kwargs
    assert call.kwargs["blocks"]  # non-empty
    assert "Weekly Keep" in call.kwargs["text"]


@pytest.mark.asyncio
async def test_dispatch_weekly_briefing_sends_placeholder_when_empty():
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from argos.brain.weekly_report import WeeklyKeepReport
    from argos.slack.briefing import dispatch_weekly_briefing

    now_utc = _dt(2026, 5, 20, 12, tzinfo=_tz.utc)
    empty_report = WeeklyKeepReport(
        total_keep_count=0,
        items=[],
        window_start=now_utc - _td(days=7),
        window_end=now_utc,
    )

    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1700.000"})
    mock_app = MagicMock()
    mock_app.client = mock_client

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.build_weekly_keep_report",
        AsyncMock(return_value=empty_report),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        result = await dispatch_weekly_briefing(channel="C999")

    # Empty portfolio MUST still send one message (skip 금지 per spec).
    assert result == "1700.000"
    assert len(mock_client.chat_postMessage.await_args_list) == 1


@pytest.mark.asyncio
async def test_dispatch_weekly_briefing_uses_configured_channel_when_none():
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from argos.brain.weekly_report import WeeklyKeepReport
    from argos.slack.briefing import dispatch_weekly_briefing

    now_utc = _dt(2026, 5, 20, 12, tzinfo=_tz.utc)
    empty_report = WeeklyKeepReport(
        total_keep_count=0,
        items=[],
        window_start=now_utc - _td(days=7),
        window_end=now_utc,
    )

    mock_client = AsyncMock()
    mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1700.111"})
    mock_app = MagicMock()
    mock_app.client = mock_client

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.build_weekly_keep_report",
        AsyncMock(return_value=empty_report),
    ), patch(
        "argos.slack.briefing.build_app", return_value=mock_app
    ), patch(
        "argos.slack.briefing.settings"
    ) as mock_settings:
        mock_settings.user.slack.channel_id = "C_DEFAULT"
        await dispatch_weekly_briefing()  # no channel override

    call = mock_client.chat_postMessage.await_args_list[0]
    assert call.kwargs["channel"] == "C_DEFAULT"
