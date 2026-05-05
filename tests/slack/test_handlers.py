from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.slack.handlers.deep_dive import handle_deep_dive
from argos.slack.handlers.keep import handle_keep
from argos.slack.handlers.pass_ import handle_pass


def _make_body(tech_id: uuid.UUID) -> dict:
    return {"actions": [{"value": str(tech_id)}]}


@pytest.mark.asyncio
async def test_keep_ack_called_first(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("argos.slack.handlers.keep.AsyncSessionLocal", return_value=mock_ctx):
        await handle_keep(mock_ack, body, mock_respond)

    assert call_order[0] == "ack"
    mock_ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_keep_persists_user_asset_with_keep_status(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)
    added_assets = []

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = lambda asset: added_assets.append(asset)
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("argos.slack.handlers.keep.AsyncSessionLocal", return_value=mock_ctx):
        await handle_keep(mock_ack, body, mock_respond)

    assert len(added_assets) == 1
    from argos.models.user_asset import AssetStatus
    assert added_assets[0].status == AssetStatus.KEEP
    assert added_assets[0].tech_id == tech_id


@pytest.mark.asyncio
async def test_pass_ack_called_first(tech_id, mock_ack, mock_respond):
    body = _make_body(tech_id)
    call_order = []
    mock_ack.side_effect = lambda: call_order.append("ack")
    mock_respond.side_effect = lambda *a, **kw: call_order.append("respond")

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

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


@pytest.mark.asyncio
async def test_keep_respond_is_ephemeral_and_preserves_original(
    tech_id, mock_ack, mock_respond
):
    body = _make_body(tech_id)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("argos.slack.handlers.keep.AsyncSessionLocal", return_value=mock_ctx):
        await handle_keep(mock_ack, body, mock_respond)

    mock_respond.assert_awaited_once()
    _assert_ephemeral_no_replace(mock_respond.call_args)


@pytest.mark.asyncio
async def test_pass_respond_is_ephemeral_and_preserves_original(
    tech_id, mock_ack, mock_respond
):
    body = _make_body(tech_id)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("argos.slack.handlers.pass_.AsyncSessionLocal", return_value=mock_ctx):
        await handle_pass(mock_ack, body, mock_respond)

    mock_respond.assert_awaited_once()
    _assert_ephemeral_no_replace(mock_respond.call_args)


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
