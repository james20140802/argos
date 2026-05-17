"""Tests for the Slack `/argos add <URL>` slash command handler (ARG-110)."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.crawler.add_url import AddUrlResult, AddUrlStatus
from argos.slack.blocks import (
    build_add_url_processing_blocks,
    build_add_url_result_blocks,
)
from argos.slack.handlers.add_url import (
    handle_argos_slash_command,
    parse_add_command,
)


# ---------------------------------------------------------------------------
# Parse helper
# ---------------------------------------------------------------------------


def test_parse_add_command_extracts_single_url():
    assert parse_add_command("add https://example.com/a") == [
        "https://example.com/a"
    ]


def test_parse_add_command_extracts_multiple_urls():
    assert parse_add_command(
        "add https://a.test/1 https://a.test/2"
    ) == ["https://a.test/1", "https://a.test/2"]


def test_parse_add_command_handles_extra_whitespace():
    assert parse_add_command("add   https://a.test/1   https://a.test/2") == [
        "https://a.test/1",
        "https://a.test/2",
    ]


def test_parse_add_command_returns_empty_without_keyword():
    assert parse_add_command("portfolio") == []


def test_parse_add_command_returns_empty_with_no_urls():
    assert parse_add_command("add") == []


def test_parse_add_command_strips_slack_link_brackets():
    """Slack wraps URLs in <...>; the parser must strip them."""
    # Plain URL inside angle brackets.
    assert parse_add_command("add <https://example.com/a>") == [
        "https://example.com/a"
    ]
    # Display-text variant: <url|displayed>
    assert parse_add_command("add <https://example.com/a|example.com/a>") == [
        "https://example.com/a"
    ]


def test_parse_add_command_is_case_insensitive_on_keyword():
    assert parse_add_command("ADD https://a.test/1") == ["https://a.test/1"]


# ---------------------------------------------------------------------------
# Block Kit builder
# ---------------------------------------------------------------------------


def test_build_add_url_result_blocks_created():
    new_id = uuid.uuid4()
    result = AddUrlResult(
        url="https://example.com/a",
        status=AddUrlStatus.CREATED,
        tech_item_id=new_id,
    )
    blocks = build_add_url_result_blocks([result])
    # Sanity-check: at least one section block carrying the URL and status.
    serialized = str(blocks)
    assert "https://example.com/a" in serialized
    assert "created" in serialized.lower() or "추가" in serialized


def test_build_add_url_result_blocks_duplicate_mentions_existing_id():
    existing = uuid.uuid4()
    result = AddUrlResult(
        url="https://example.com/a",
        status=AddUrlStatus.DUPLICATE,
        tech_item_id=existing,
    )
    blocks = build_add_url_result_blocks([result])
    serialized = str(blocks)
    # Display the existing id (or a short prefix) so the user can find it.
    assert str(existing)[:8] in serialized


def test_build_add_url_result_blocks_rejected_includes_reason():
    result = AddUrlResult(
        url="https://example.com/a",
        status=AddUrlStatus.REJECTED,
        reason="robots.txt disallows",
    )
    blocks = build_add_url_result_blocks([result])
    serialized = str(blocks)
    assert "robots" in serialized.lower()


def test_build_add_url_result_blocks_multiple_results():
    results = [
        AddUrlResult("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        AddUrlResult("https://b", AddUrlStatus.REJECTED, reason="bad"),
    ]
    blocks = build_add_url_result_blocks(results)
    serialized = str(blocks)
    assert "https://a" in serialized
    assert "https://b" in serialized


def test_build_add_url_result_blocks_empty_list_renders_message():
    blocks = build_add_url_result_blocks([])
    # Even with no results we should produce something coherent for Slack.
    assert len(blocks) >= 1


def test_build_add_url_result_blocks_respects_slack_50_block_cap():
    """Large `/argos add` inputs must never exceed Slack's 50-block limit
    (Codex review on PR #67). 30 URL results would naïvely emit
    1 header + 30 * 2 = 61 blocks.
    """
    from argos.slack.blocks import SLACK_MAX_BLOCKS

    results = [
        AddUrlResult(
            url=f"https://example.com/{i}",
            status=AddUrlStatus.CREATED,
            tech_item_id=uuid.uuid4(),
        )
        for i in range(30)
    ]
    blocks = build_add_url_result_blocks(results)
    assert len(blocks) <= SLACK_MAX_BLOCKS


def test_build_add_url_result_blocks_appends_truncation_notice_when_over_cap():
    """When the result list overflows the block cap, the last block should be
    a context block noting how many results were omitted.
    """
    from argos.slack.blocks import SLACK_MAX_BLOCKS

    total = 40
    results = [
        AddUrlResult(
            url=f"https://example.com/{i}",
            status=AddUrlStatus.CREATED,
            tech_item_id=uuid.uuid4(),
        )
        for i in range(total)
    ]
    blocks = build_add_url_result_blocks(results)
    assert len(blocks) <= SLACK_MAX_BLOCKS
    assert blocks[-1]["type"] == "context"
    notice_text = blocks[-1]["elements"][0]["text"]
    # Notice should reference the hidden count explicitly. Visible result
    # blocks each take 2 blocks (section + divider) minus the trailing
    # divider drop, so we recompute the expected hidden count.
    visible_sections = sum(1 for b in blocks if b.get("type") == "section")
    expected_hidden = total - visible_sections
    assert expected_hidden >= 1
    assert str(expected_hidden) in notice_text


def test_build_add_url_result_blocks_long_url_stays_under_section_limit():
    """A pathologically long but valid URL (e.g., tracking-heavy link) must
    not blow past Slack's 3000-char section text limit (Codex review on
    PR #67 follow-up). Previously the URL was duplicated as `<url|url>`,
    which trivially exceeded the cap for any URL longer than ~1500 chars
    and triggered `invalid_blocks`.
    """
    from argos.slack.blocks import SLACK_SECTION_TEXT_LIMIT

    long_url = "https://example.com/?" + ("q=" + "a" * 50 + "&") * 100  # >2500 chars
    assert len(long_url) > 2000
    result = AddUrlResult(
        url=long_url,
        status=AddUrlStatus.CREATED,
        tech_item_id=uuid.uuid4(),
    )

    blocks = build_add_url_result_blocks([result])

    # Find the section block carrying the URL.
    section_blocks = [b for b in blocks if b.get("type") == "section"]
    assert section_blocks, "expected at least one section block"
    for block in section_blocks:
        text = block["text"]["text"]
        assert len(text) <= SLACK_SECTION_TEXT_LIMIT, (
            f"section text exceeds Slack limit: {len(text)} > {SLACK_SECTION_TEXT_LIMIT}"
        )


def test_build_add_url_processing_blocks_clamps_long_single_url():
    """A pathologically long single URL must not push the interim
    'processing…' section text past Slack's 3000-char limit; otherwise
    /argos add fails with invalid_blocks before the background task runs.
    """
    from argos.slack.blocks import SLACK_SECTION_TEXT_LIMIT

    long_url = "https://example.com/?" + ("q=" + "a" * 50 + "&") * 100
    assert len(long_url) > SLACK_SECTION_TEXT_LIMIT

    blocks = build_add_url_processing_blocks([long_url])

    assert len(blocks) == 1
    text = blocks[0]["text"]["text"]
    assert len(text) <= SLACK_SECTION_TEXT_LIMIT, (
        f"processing section text exceeds Slack limit: {len(text)} > {SLACK_SECTION_TEXT_LIMIT}"
    )


def test_build_add_url_result_blocks_no_truncation_notice_under_cap():
    """A handful of results should render without any truncation notice."""
    results = [
        AddUrlResult(
            url=f"https://example.com/{i}",
            status=AddUrlStatus.CREATED,
            tech_item_id=uuid.uuid4(),
        )
        for i in range(3)
    ]
    blocks = build_add_url_result_blocks(results)
    assert all(b.get("type") != "context" for b in blocks)


# ---------------------------------------------------------------------------
# Slash command handler — ack + dispatch
# ---------------------------------------------------------------------------


def _make_session_ctx():
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return session, ctx


@pytest.mark.asyncio
async def test_slash_command_acks_first(mock_ack, mock_respond):
    """Ack must be the very first await — Slack requires <3s response."""
    call_order: list[str] = []
    mock_ack.side_effect = lambda: call_order.append("ack")

    async def _respond(*a, **kw):
        call_order.append("respond")

    mock_respond.side_effect = _respond

    command = {"text": "add https://example.com/a"}

    fake_result = AddUrlResult(
        "https://example.com/a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()
    )
    session, ctx = _make_session_ctx()

    with (
        patch(
            "argos.slack.handlers.add_url.AsyncSessionLocal", return_value=ctx
        ),
        patch(
            "argos.slack.handlers.add_url.add_url_service",
            new=AsyncMock(return_value=fake_result),
        ),
    ):
        await handle_argos_slash_command(mock_ack, command, mock_respond)
        # Let the background task finish so the test sees its respond call.
        await asyncio.sleep(0)
        # Drain any pending tasks.
        for _ in range(5):
            await asyncio.sleep(0)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_slash_command_routes_add_to_add_handler(mock_ack, mock_respond):
    command = {"text": "add https://example.com/a"}
    fake_result = AddUrlResult(
        "https://example.com/a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()
    )
    session, ctx = _make_session_ctx()
    add_mock = AsyncMock(return_value=fake_result)

    with (
        patch(
            "argos.slack.handlers.add_url.AsyncSessionLocal", return_value=ctx
        ),
        patch("argos.slack.handlers.add_url.add_url_service", add_mock),
    ):
        await handle_argos_slash_command(mock_ack, command, mock_respond)
        for _ in range(10):
            await asyncio.sleep(0)

    add_mock.assert_awaited()
    awaited_url = add_mock.await_args.args[0]
    assert awaited_url == "https://example.com/a"


@pytest.mark.asyncio
async def test_slash_command_falls_back_to_portfolio_without_add(
    mock_ack, mock_respond
):
    """`/argos` with no add keyword should delegate to the portfolio handler."""
    command = {"text": ""}

    portfolio_mock = AsyncMock()
    with patch(
        "argos.slack.handlers.add_url.handle_portfolio_command",
        portfolio_mock,
    ):
        await handle_argos_slash_command(mock_ack, command, mock_respond)

    portfolio_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_slash_command_responds_with_help_when_add_has_no_urls(
    mock_ack, mock_respond
):
    command = {"text": "add"}

    add_mock = AsyncMock()
    with patch("argos.slack.handlers.add_url.add_url_service", add_mock):
        await handle_argos_slash_command(mock_ack, command, mock_respond)
        for _ in range(5):
            await asyncio.sleep(0)

    add_mock.assert_not_awaited()
    mock_respond.assert_awaited()
    # The user-facing message should mention "URL" so they understand.
    args, kwargs = mock_respond.call_args
    blob = str(args) + str(kwargs)
    assert "url" in blob.lower()


@pytest.mark.asyncio
async def test_slash_command_handles_multiple_urls(mock_ack, mock_respond):
    command = {"text": "add https://a https://b"}
    results = [
        AddUrlResult("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        AddUrlResult("https://b", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
    ]
    session, ctx = _make_session_ctx()
    add_mock = AsyncMock(side_effect=results)

    with (
        patch(
            "argos.slack.handlers.add_url.AsyncSessionLocal", return_value=ctx
        ),
        patch("argos.slack.handlers.add_url.add_url_service", add_mock),
    ):
        await handle_argos_slash_command(mock_ack, command, mock_respond)
        for _ in range(10):
            await asyncio.sleep(0)

    assert add_mock.await_count == 2
    urls = [call.args[0] for call in add_mock.await_args_list]
    assert urls == ["https://a", "https://b"]


@pytest.mark.asyncio
async def test_slash_command_logs_and_replies_on_service_exception(
    mock_ack, mock_respond, caplog
):
    """If add_url_service raises, surface a friendly message + log details."""
    command = {"text": "add https://a"}
    session, ctx = _make_session_ctx()
    add_mock = AsyncMock(side_effect=RuntimeError("kaboom"))

    with (
        patch(
            "argos.slack.handlers.add_url.AsyncSessionLocal", return_value=ctx
        ),
        patch("argos.slack.handlers.add_url.add_url_service", add_mock),
    ):
        await handle_argos_slash_command(mock_ack, command, mock_respond)
        for _ in range(10):
            await asyncio.sleep(0)

    # The user gets at least one respond call (the friendly error).
    assert mock_respond.await_count >= 1
    # The exception detail is in logs, not the user-facing message.
    log_text = " ".join(rec.message for rec in caplog.records)
    assert "kaboom" in log_text or "kaboom" in " ".join(
        rec.exc_text or "" for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_slash_command_sends_processing_message_then_results(
    mock_ack, mock_respond
):
    """The handler should send a 'processing…' interim message, then results."""
    command = {"text": "add https://a"}
    session, ctx = _make_session_ctx()
    result = AddUrlResult(
        "https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()
    )
    add_mock = AsyncMock(return_value=result)

    with (
        patch(
            "argos.slack.handlers.add_url.AsyncSessionLocal", return_value=ctx
        ),
        patch("argos.slack.handlers.add_url.add_url_service", add_mock),
    ):
        await handle_argos_slash_command(mock_ack, command, mock_respond)
        for _ in range(10):
            await asyncio.sleep(0)

    # At least 2 respond calls: processing + final.
    assert mock_respond.await_count >= 2
