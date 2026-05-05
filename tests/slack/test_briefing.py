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
async def test_dispatch_skips_when_all_categories_empty():
    items_by_category = {CategoryType.MAINSTREAM: [], CategoryType.ALPHA: []}
    mock_client, mock_app, session_ctx = _patch_dispatch(items_by_category)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        result = await dispatch_daily_briefing(channel="C999")

    assert result is None
    mock_client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_posts_header_then_threaded_items():
    items_by_category = {
        CategoryType.MAINSTREAM: [_make_item("Stream-A"), _make_item("Stream-B")],
        CategoryType.ALPHA: [_make_item("Alpha-A")],
    }
    mock_client, mock_app, session_ctx = _patch_dispatch(items_by_category)

    with patch(
        "argos.slack.briefing.AsyncSessionLocal", return_value=session_ctx
    ), patch(
        "argos.slack.briefing.fetch_today_briefing",
        AsyncMock(return_value=items_by_category),
    ), patch("argos.slack.briefing.build_app", return_value=mock_app):
        result = await dispatch_daily_briefing(channel="C999")

    assert result == "1700000000.001"
    calls = mock_client.chat_postMessage.await_args_list
    # 1 header + 2 category headers + 3 item replies
    assert len(calls) == 1 + 2 + 3

    header_call = calls[0]
    assert "thread_ts" not in header_call.kwargs
    assert header_call.kwargs["channel"] == "C999"

    for follow_up in calls[1:]:
        assert follow_up.kwargs["thread_ts"] == "1700000000.001"
        assert follow_up.kwargs["reply_broadcast"] is True
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
