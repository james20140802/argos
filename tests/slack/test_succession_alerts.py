"""ARG-104: Slack succession alert blocks + post_track_update dispatcher.

Covers:
- `build_succession_alert_blocks` block-structure shape.
- `post_track_update` posts one message per alert and writes a track_history
  row marking the alert as delivered.
- Slack failure on one alert does not stop the rest and does not write
  history for the failing one.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.models.tech_succession import RelationType
from argos.models.track_history import TrackHistory
from argos.slack.blocks import build_succession_alert_blocks
from argos.slack.services.track_check import (
    SUCCESSION_ALERTED,
    SuccessionAlert,
    post_track_update,
)


def _alert(**overrides) -> SuccessionAlert:
    base = {
        "user_asset_id": uuid.uuid4(),
        "predecessor_title": "Old Tech",
        "successor_title": "New Tech",
        "relation_type": RelationType.REPLACE,
    }
    base.update(overrides)
    return SuccessionAlert(**base)


# ─── build_succession_alert_blocks ──────────────────────────────────────────


def test_build_succession_alert_blocks_basic_shape():
    alert = _alert()
    blocks = build_succession_alert_blocks(alert)

    assert isinstance(blocks, list)
    assert len(blocks) >= 1
    section = blocks[0]
    assert section["type"] == "section"
    assert section["text"]["type"] == "mrkdwn"
    text = section["text"]["text"]

    # Required substrings per spec.
    assert "⚠️" in text
    assert "Keep" in text
    assert "Old Tech" in text
    assert "New Tech" in text
    assert "Replace" in text  # relation_type value


def test_build_succession_alert_blocks_includes_enhance_label():
    alert = _alert(relation_type=RelationType.ENHANCE)
    blocks = build_succession_alert_blocks(alert)
    assert "Enhance" in blocks[0]["text"]["text"]


def test_build_succession_alert_blocks_includes_fork_label():
    alert = _alert(relation_type=RelationType.FORK)
    blocks = build_succession_alert_blocks(alert)
    assert "Fork" in blocks[0]["text"]["text"]


def test_build_succession_alert_blocks_emboldens_titles():
    alert = _alert(predecessor_title="OldA", successor_title="NewB")
    text = build_succession_alert_blocks(alert)[0]["text"]["text"]
    # Both titles must be wrapped in mrkdwn bold (*…*).
    assert "*OldA*" in text
    assert "*NewB*" in text


# ─── post_track_update ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_track_update_posts_message_and_writes_history():
    alert = _alert()
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})

    added: list = []
    session = AsyncMock()
    session.add = lambda obj: added.append(obj)
    session.flush = AsyncMock()

    await post_track_update(app, "C123", [alert], session)

    # One chat_postMessage to the configured channel.
    app.client.chat_postMessage.assert_awaited_once()
    kwargs = app.client.chat_postMessage.await_args.kwargs
    assert kwargs["channel"] == "C123"
    assert isinstance(kwargs["blocks"], list)
    assert kwargs.get("text"), "Slack requires fallback text"

    # Exactly one TrackHistory row written.
    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert len(history_rows) == 1
    row = history_rows[0]
    assert row.user_asset_id == alert.user_asset_id
    assert row.changed_to == SUCCESSION_ALERTED
    # changed_from is NOT NULL on the model; convention: 'Keep'.
    assert row.changed_from == "Keep"


@pytest.mark.asyncio
async def test_post_track_update_empty_alerts_is_noop():
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_postMessage = AsyncMock()
    session = AsyncMock()

    await post_track_update(app, "C123", [], session)

    app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_track_update_continues_on_partial_failure():
    """If Slack fails on the first alert, the second still gets sent and only
    successful sends write history."""
    a1 = _alert()
    a2 = _alert()
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_postMessage = AsyncMock(
        side_effect=[RuntimeError("slack down"), {"ok": True, "ts": "2"}]
    )

    added: list = []
    session = AsyncMock()
    session.add = lambda obj: added.append(obj)
    session.flush = AsyncMock()

    await post_track_update(app, "C123", [a1, a2], session)

    assert app.client.chat_postMessage.await_count == 2
    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert len(history_rows) == 1
    assert history_rows[0].user_asset_id == a2.user_asset_id


@pytest.mark.asyncio
async def test_post_track_update_writes_history_even_if_flush_unused():
    """The dispatcher should add() but not require the caller to flush — flush
    timing belongs to the caller (CLI commits the session)."""
    alert = _alert()
    app = MagicMock()
    app.client = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})

    added: list = []
    session = AsyncMock()
    session.add = lambda obj: added.append(obj)
    session.flush = AsyncMock()

    await post_track_update(app, "C123", [alert], session)

    history_rows = [o for o in added if isinstance(o, TrackHistory)]
    assert len(history_rows) == 1
