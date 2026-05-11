from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.slack.handlers.deep_dive import handle_deep_dive
from argos.slack.handlers.keep import handle_keep
from argos.slack.handlers.pass_ import handle_pass
from argos.slack.handlers.portfolio import handle_portfolio_command
from argos.slack.handlers.untrack import handle_untrack


def _make_body(tech_id: uuid.UUID) -> dict:
    return {"actions": [{"value": str(tech_id)}]}


def _make_insert_session(inserted_id: uuid.UUID | None = None) -> tuple[AsyncMock, MagicMock]:
    """Build a session whose INSERT ... ON CONFLICT returns `inserted_id`.

    When `inserted_id` is a UUID, the upsert took the CREATED branch and the
    follow-up SELECT is never executed; passing None here would force the
    handler into the lock-and-read path which these tests don't exercise.
    """
    if inserted_id is None:
        inserted_id = uuid.uuid4()
    mock_session = AsyncMock()
    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = inserted_id
    mock_session.execute = AsyncMock(return_value=insert_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_session, mock_ctx


@pytest.mark.asyncio
async def test_keep_ack_called_first(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    _, mock_ctx = _make_insert_session()

    with patch("argos.slack.handlers.keep.AsyncSessionLocal", return_value=mock_ctx):
        await handle_keep(mock_ack, body, mock_respond)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_keep_inserts_user_asset_with_keep_status(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)

    mock_session, mock_ctx = _make_insert_session()

    with patch("argos.slack.handlers.keep.AsyncSessionLocal", return_value=mock_ctx):
        await handle_keep(mock_ack, body, mock_respond)

    # Upsert path: one execute (INSERT ON CONFLICT), no follow-up select.
    assert mock_session.execute.await_count == 1
    insert_stmt = mock_session.execute.await_args.args[0]
    compiled = str(
        insert_stmt.compile(compile_kwargs={"literal_binds": True})
    )
    assert "INSERT INTO user_assets" in compiled
    assert "ON CONFLICT" in compiled
    assert str(tech_id) in compiled
    assert "Keep" in compiled


@pytest.mark.asyncio
async def test_pass_ack_called_first(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    _, mock_ctx = _make_insert_session()

    with patch("argos.slack.handlers.pass_.AsyncSessionLocal", return_value=mock_ctx):
        await handle_pass(mock_ack, body, mock_respond)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_deep_dive_ack_called_first(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    with patch("asyncio.create_task"):
        await handle_deep_dive(mock_ack, body, mock_respond)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_deep_dive_respond_called_immediately(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)

    with patch("asyncio.create_task"):
        await handle_deep_dive(mock_ack, body, mock_respond)

    mock_respond.assert_awaited_once()
    call_args = mock_respond.call_args[0][0]
    assert "심층 분석" in call_args or "분석" in call_args


@pytest.mark.asyncio
async def test_deep_dive_creates_task(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)

    with patch("asyncio.create_task") as mock_create_task:
        await handle_deep_dive(mock_ack, body, mock_respond)
        mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_keep_invalid_uuid_responds_with_error(mock_ack, mock_respond):
    body = {"actions": [{"value": "not-a-uuid"}]}
    await handle_keep(mock_ack, body, mock_respond)
    mock_ack.assert_awaited_once()
    mock_respond.assert_awaited_once()
    assert "잘못된" in mock_respond.call_args[0][0]


@pytest.mark.asyncio
async def test_pass_invalid_uuid_responds_with_error(mock_ack, mock_respond):
    body = {"actions": [{"value": "not-a-uuid"}]}
    await handle_pass(mock_ack, body, mock_respond)
    mock_ack.assert_awaited_once()
    mock_respond.assert_awaited_once()
    assert "잘못된" in mock_respond.call_args[0][0]


def _assert_ephemeral_no_replace(mock_call):
    kwargs = mock_call.kwargs
    assert kwargs.get("response_type") == "ephemeral"
    assert kwargs.get("replace_original") is False


def _make_body_with_message(tech_id: uuid.UUID, *, original_blocks: list[dict]) -> dict:
    return {
        "actions": [{"value": str(tech_id)}],
        "message": {"blocks": original_blocks},
    }


def _original_card_blocks(tech_id: uuid.UUID) -> list[dict]:
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Sample item* (trust=0.50)\nhttps://x"},
        },
        {
            "type": "actions",
            "elements": [
                {"type": "button", "action_id": "action_keep", "value": str(tech_id)},
            ],
        },
    ]


@pytest.mark.asyncio
async def test_keep_respond_replaces_card_with_keep_status_and_drops_buttons(
    tech_id, mock_ack, mock_respond
):
    from argos.slack.blocks import ITEM_STATUS_BLOCK_ID

    body = _make_body_with_message(tech_id, original_blocks=_original_card_blocks(tech_id))
    _, mock_ctx = _make_insert_session()

    with patch("argos.slack.handlers.keep.AsyncSessionLocal", return_value=mock_ctx):
        await handle_keep(mock_ack, body, mock_respond)

    mock_respond.assert_awaited_once()
    kwargs = mock_respond.call_args.kwargs
    assert kwargs.get("replace_original") is True
    new_blocks = kwargs["blocks"]
    # buttons removed, status block appended
    assert all(b.get("type") != "actions" for b in new_blocks)
    status_block = new_blocks[-1]
    assert status_block["block_id"] == ITEM_STATUS_BLOCK_ID
    assert "Keep" in status_block["elements"][0]["text"]


@pytest.mark.asyncio
async def test_pass_respond_replaces_card_with_archived_status_and_drops_buttons(
    tech_id, mock_ack, mock_respond
):
    from argos.slack.blocks import ITEM_STATUS_BLOCK_ID

    body = _make_body_with_message(tech_id, original_blocks=_original_card_blocks(tech_id))
    _, mock_ctx = _make_insert_session()

    with patch("argos.slack.handlers.pass_.AsyncSessionLocal", return_value=mock_ctx):
        await handle_pass(mock_ack, body, mock_respond)

    mock_respond.assert_awaited_once()
    kwargs = mock_respond.call_args.kwargs
    assert kwargs.get("replace_original") is True
    new_blocks = kwargs["blocks"]
    assert all(b.get("type") != "actions" for b in new_blocks)
    status_block = new_blocks[-1]
    assert status_block["block_id"] == ITEM_STATUS_BLOCK_ID
    assert "Archived" in status_block["elements"][0]["text"]


@pytest.mark.asyncio
async def test_deep_dive_respond_is_ephemeral_and_preserves_original(
    tech_id, mock_ack, mock_respond
):
    body = _make_body(tech_id)

    with patch("asyncio.create_task"):
        await handle_deep_dive(mock_ack, body, mock_respond)

    mock_respond.assert_awaited_once()
    _assert_ephemeral_no_replace(mock_respond.call_args)


@pytest.mark.asyncio
async def test_deep_dive_passes_thread_metadata_to_background_task(
    tech_id, mock_ack, mock_respond
):
    import inspect

    body = {
        "actions": [{"value": str(tech_id)}],
        "channel": {"id": "C123"},
        "message": {"ts": "1700000000.001"},
    }
    captured = {}

    def capture(coro):
        captured["locals"] = inspect.getcoroutinelocals(coro)
        coro.close()
        return MagicMock()

    mock_client = AsyncMock()
    with patch("asyncio.create_task", side_effect=capture):
        await handle_deep_dive(mock_ack, body, mock_respond, client=mock_client)

    locals_ = captured["locals"]
    assert locals_["channel_id"] == "C123"
    assert locals_["thread_ts"] == "1700000000.001"
    assert locals_["client"] is mock_client


# ---------------------------------------------------------------------------
# handle_portfolio_command tests
# ---------------------------------------------------------------------------


def _make_portfolio_session_ctx(assets: list) -> tuple[AsyncMock, MagicMock]:
    """Build an async session context mock that returns `assets` from execute."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = assets
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_session, mock_ctx


def _make_asset_item_pair(tech_id_val: uuid.UUID):
    item = SimpleNamespace(
        id=tech_id_val,
        title="TestTech",
        source_url="https://example.com/test",
    )
    asset = MagicMock()
    asset.updated_at = datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone.utc)
    asset.last_monitored_at = None
    return (asset, item)


@pytest.mark.asyncio
async def test_portfolio_command_ack_called_first(tech_id, mock_ack, mock_respond):
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    _, mock_ctx = _make_portfolio_session_ctx([_make_asset_item_pair(tech_id)])

    with patch("argos.slack.handlers.portfolio.AsyncSessionLocal", return_value=mock_ctx):
        await handle_portfolio_command(mock_ack, {}, mock_respond)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_portfolio_command_responds_ephemerally_with_assets(tech_id, mock_ack, mock_respond):
    pair = _make_asset_item_pair(tech_id)
    _, mock_ctx = _make_portfolio_session_ctx([pair])

    with patch("argos.slack.handlers.portfolio.AsyncSessionLocal", return_value=mock_ctx):
        await handle_portfolio_command(mock_ack, {}, mock_respond)

    mock_respond.assert_awaited_once()
    kwargs = mock_respond.call_args.kwargs
    assert kwargs.get("response_type") == "ephemeral"
    assert "blocks" in kwargs
    # Should have header block
    header = kwargs["blocks"][0]
    assert header["type"] == "header"


@pytest.mark.asyncio
async def test_portfolio_command_responds_with_empty_state_when_no_assets(mock_ack, mock_respond):
    _, mock_ctx = _make_portfolio_session_ctx([])

    with patch("argos.slack.handlers.portfolio.AsyncSessionLocal", return_value=mock_ctx):
        await handle_portfolio_command(mock_ack, {}, mock_respond)

    mock_respond.assert_awaited_once()
    kwargs = mock_respond.call_args.kwargs
    all_text = str(kwargs["blocks"])
    assert "Keep한 기술이 없습니다" in all_text


# ---------------------------------------------------------------------------
# handle_untrack tests
# ---------------------------------------------------------------------------


def _make_untrack_body(tech_id: uuid.UUID) -> dict:
    return {"actions": [{"value": str(tech_id)}]}


def _make_untrack_session_ctx(
    inserted_id: uuid.UUID | None = None,
    portfolio_assets: list | None = None,
) -> tuple[AsyncMock, MagicMock]:
    """Build a session that handles both the transition INSERT and portfolio SELECT."""
    if inserted_id is None:
        inserted_id = uuid.uuid4()
    if portfolio_assets is None:
        portfolio_assets = []

    mock_session = AsyncMock()
    call_count = 0

    async def _execute(stmt):
        nonlocal call_count
        mock_result = MagicMock()
        if call_count == 0:
            # First call is INSERT ... ON CONFLICT from transition_asset
            mock_result.scalar_one_or_none.return_value = inserted_id
        else:
            # Subsequent calls are SELECT for fetch_user_portfolio
            mock_result.all.return_value = portfolio_assets
        call_count += 1
        return mock_result

    mock_session.execute = _execute
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_session, mock_ctx


@pytest.mark.asyncio
async def test_untrack_ack_called_first(tech_id, mock_ack, mock_respond):
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    _, mock_ctx = _make_untrack_session_ctx()

    with patch("argos.slack.handlers.untrack.AsyncSessionLocal", return_value=mock_ctx):
        await handle_untrack(mock_ack, _make_untrack_body(tech_id), mock_respond)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_untrack_invalid_uuid_responds_with_error(mock_ack, mock_respond):
    body = {"actions": [{"value": "not-a-uuid"}]}
    await handle_untrack(mock_ack, body, mock_respond)
    mock_ack.assert_awaited_once()
    mock_respond.assert_awaited_once()
    assert "잘못된" in mock_respond.call_args[0][0]


@pytest.mark.asyncio
async def test_untrack_success_rerenders_portfolio_without_removed_asset(
    tech_id, mock_ack, mock_respond
):
    # After untrack, portfolio is empty
    _, mock_ctx = _make_untrack_session_ctx(portfolio_assets=[])

    with patch("argos.slack.handlers.untrack.AsyncSessionLocal", return_value=mock_ctx):
        await handle_untrack(mock_ack, _make_untrack_body(tech_id), mock_respond)

    mock_respond.assert_awaited_once()
    kwargs = mock_respond.call_args.kwargs
    assert kwargs.get("replace_original") is True
    # Should show empty state
    all_text = str(kwargs["blocks"])
    assert "Keep한 기술이 없습니다" in all_text


@pytest.mark.asyncio
async def test_untrack_commits_session(tech_id, mock_ack, mock_respond):
    mock_session, mock_ctx = _make_untrack_session_ctx()

    with patch("argos.slack.handlers.untrack.AsyncSessionLocal", return_value=mock_ctx):
        await handle_untrack(mock_ack, _make_untrack_body(tech_id), mock_respond)

    mock_session.commit.assert_awaited_once()
