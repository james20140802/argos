from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.models.tech_succession import RelationType
from argos.slack.services.track_check import SuccessionAlert, check_succession


def _make_session(rows: list[tuple]) -> AsyncMock:
    """Mock an AsyncSession.execute that returns a Result whose ``all()`` is rows."""
    result = MagicMock()
    result.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_check_succession_empty_input_returns_empty():
    session = AsyncMock()
    alerts = await check_succession(session, [])
    assert alerts == []
    # No query should be issued when there are no candidate IDs.
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_succession_returns_matched_keep_alerts():
    successor_id = uuid.uuid4()
    user_asset_id = uuid.uuid4()
    # Row schema: (user_asset_id, predecessor_title, successor_title, relation_type)
    rows = [
        (
            user_asset_id,
            "Old Tech",
            "New Tech",
            RelationType.REPLACE,
        )
    ]
    session = _make_session(rows)

    alerts = await check_succession(session, [successor_id])

    assert len(alerts) == 1
    alert = alerts[0]
    assert isinstance(alert, SuccessionAlert)
    assert alert.user_asset_id == user_asset_id
    assert alert.predecessor_title == "Old Tech"
    assert alert.successor_title == "New Tech"
    assert alert.relation_type is RelationType.REPLACE


@pytest.mark.asyncio
async def test_check_succession_no_matching_predecessor_returns_empty():
    """If no Keep-ed predecessor matches, the query returns no rows."""
    successor_id = uuid.uuid4()
    session = _make_session([])

    alerts = await check_succession(session, [successor_id])

    assert alerts == []


@pytest.mark.asyncio
async def test_check_succession_filters_already_alerted_via_query():
    """The SQL itself excludes already-alerted user_assets, so the result set
    only contains new alerts. We verify the function returns whatever rows
    the DB hands back, and that LEFT JOIN/NOT EXISTS subquery referencing
    track_history is part of the executed statement."""
    successor_id = uuid.uuid4()
    # DB returns no rows because the only candidate has already been alerted.
    session = _make_session([])

    alerts = await check_succession(session, [successor_id])
    assert alerts == []

    # Verify a SELECT was issued and that the rendered SQL references both
    # tech_succession and a track_history dedup predicate.
    session.execute.assert_awaited()
    executed_stmt = session.execute.await_args.args[0]
    rendered = str(executed_stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "tech_succession" in rendered
    assert "track_history" in rendered


@pytest.mark.asyncio
async def test_check_succession_multiple_alerts_preserves_order():
    successor_id = uuid.uuid4()
    asset_a = uuid.uuid4()
    asset_b = uuid.uuid4()
    rows = [
        (asset_a, "A old", "A new", RelationType.ENHANCE),
        (asset_b, "B old", "B new", RelationType.FORK),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session, [successor_id])

    assert [a.user_asset_id for a in alerts] == [asset_a, asset_b]
    assert [a.relation_type for a in alerts] == [
        RelationType.ENHANCE,
        RelationType.FORK,
    ]
