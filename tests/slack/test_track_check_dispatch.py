"""ARG-116: unit tests for post_signal_update Slack dispatcher.

Covers:
- Happy path: one chat_postMessage per match, one TrackHistory row written,
  last_monitored_at updated.
- Empty matches list → no-op.
- Partial failure: Slack exception on first match does not stop the second;
  only the successful send writes history and updates last_monitored_at.
- Block Kit block structure from build_signal_match_blocks.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.models.track_history import TrackHistory
from argos.slack.blocks import build_signal_match_blocks
from argos.slack.services.track_check import (
    SIGNAL_MATCHED,
    SignalMatch,
    post_signal_update,
)


def _match(**overrides) -> SignalMatch:
    base = {
        "user_asset_id": uuid.uuid4(),
        "keep_item_id": uuid.uuid4(),
        "keep_item_title": "Keep Tech",
        "new_item_id": uuid.uuid4(),
        "new_item_title": "New Signal",
        "new_item_url": "https://example.com/new",
        "similarity_score": 0.92,
    }
    base.update(overrides)
    return SignalMatch(**base)


def _make_session() -> tuple[AsyncMock, list]:
    """Return (session_mock, added_objects_list)."""
    added: list = []
    session = AsyncMock()
    session.add = lambda obj: added.append(obj)
    # session.execute is used by the last_monitored_at update
    session.execute = AsyncMock()
    return session, added


# ---------------------------------------------------------------------------
# build_signal_match_blocks
# ---------------------------------------------------------------------------


def test_build_signal_match_blocks_basic_shape():
    m = _match()
    blocks = build_signal_match_blocks(m)

    assert isinstance(blocks, list)
    assert len(blocks) >= 1
    section = blocks[0]
    assert section["type"] == "section"
    assert section["text"]["type"] == "mrkdwn"
    text = section["text"]["text"]

    assert "🔭" in text
    assert "Keep Tech" in text
    assert "New Signal" in text
    assert "https://example.com/new" in text
    assert "92%" in text  # similarity_score formatted as percentage


def test_build_signal_match_blocks_emboldens_titles():
    m = _match(keep_item_title="OldKeep", new_item_title="FreshSignal")
    text = build_signal_match_blocks(m)[0]["text"]["text"]
    assert "*OldKeep*" in text
    assert "*FreshSignal*" in text


def test_build_signal_match_blocks_long_text_clamped():
    """Titles near Slack's 3000-char limit should be clamped, not raise."""
    long_title = "A" * 600
    m = _match(keep_item_title=long_title, new_item_title=long_title)
    blocks = build_signal_match_blocks(m)
    text = blocks[0]["text"]["text"]
    assert len(text) <= 3000


# ---------------------------------------------------------------------------
# post_signal_update — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_signal_update_posts_one_message_per_match():
    m = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})
    session, added = _make_session()

    await post_signal_update(app, "C999", [m], session)

    app.client.chat_postMessage.assert_awaited_once()
    kwargs = app.client.chat_postMessage.await_args.kwargs
    assert kwargs["channel"] == "C999"
    assert isinstance(kwargs["blocks"], list)
    assert kwargs.get("text"), "Slack requires fallback text"
    assert "🔭" in kwargs["text"]


@pytest.mark.asyncio
async def test_post_signal_update_writes_track_history_row():
    m = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})
    session, added = _make_session()

    await post_signal_update(app, "C999", [m], session)

    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert len(history_rows) == 1
    row = history_rows[0]
    assert row.user_asset_id == m.user_asset_id
    assert row.changed_to == SIGNAL_MATCHED
    # changed_from encodes the new_item_id as a string
    assert row.changed_from == str(m.new_item_id)


@pytest.mark.asyncio
async def test_post_signal_update_updates_last_monitored_at():
    m = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})
    session, added = _make_session()

    await post_signal_update(app, "C999", [m], session)

    # session.execute should be called for the UPDATE user_assets SET last_monitored_at
    session.execute.assert_awaited()


@pytest.mark.asyncio
async def test_post_signal_update_sends_multiple_matches():
    m1 = _match()
    m2 = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(
        side_effect=[{"ok": True, "ts": "1"}, {"ok": True, "ts": "2"}]
    )
    session, added = _make_session()

    await post_signal_update(app, "C999", [m1, m2], session)

    assert app.client.chat_postMessage.await_count == 2
    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert len(history_rows) == 2


# ---------------------------------------------------------------------------
# post_signal_update — no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_signal_update_empty_matches_is_noop():
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock()
    session, added = _make_session()

    await post_signal_update(app, "C999", [], session)

    app.client.chat_postMessage.assert_not_awaited()
    assert added == []


# ---------------------------------------------------------------------------
# post_signal_update — failure handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_signal_update_continues_on_partial_failure():
    """Slack failure on the first match must not prevent the second from being processed."""
    m1 = _match()
    m2 = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(
        side_effect=[RuntimeError("slack down"), {"ok": True, "ts": "2"}]
    )
    session, added = _make_session()

    await post_signal_update(app, "C999", [m1, m2], session)

    assert app.client.chat_postMessage.await_count == 2
    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    # Only the successful match (m2) writes a history row
    assert len(history_rows) == 1
    assert history_rows[0].user_asset_id == m2.user_asset_id


@pytest.mark.asyncio
async def test_post_signal_update_no_db_sideeffect_on_failure():
    """When Slack fails, no TrackHistory row and no last_monitored_at update."""
    m = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(side_effect=RuntimeError("slack down"))
    session, added = _make_session()

    await post_signal_update(app, "C999", [m], session)

    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert history_rows == []
    # session.execute should NOT have been called (no UPDATE)
    session.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# Dedup encoding correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_track_history_changed_from_fits_in_string50():
    """str(uuid) is exactly 36 chars — must fit in String(50)."""
    m = _match()
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})
    session, added = _make_session()

    await post_signal_update(app, "C999", [m], session)

    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert len(history_rows) == 1
    changed_from = history_rows[0].changed_from
    changed_to = history_rows[0].changed_to
    assert len(changed_from) <= 50, f"changed_from too long: {len(changed_from)} chars"
    assert len(changed_to) <= 50, f"changed_to too long: {len(changed_to)} chars"
    # Specific lengths: UUID str = 36, 'signal_matched' = 14
    assert len(changed_from) == 36
    assert changed_to == "signal_matched"
