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
async def test_check_succession_explicit_empty_list_returns_empty():
    """An explicit empty list means "look at nothing" — short-circuit."""
    session = AsyncMock()
    alerts = await check_succession(session, [])
    assert alerts == []
    # No query should be issued when the caller explicitly passed [].
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_succession_none_scans_all_unalerted_rows():
    """new_item_ids=None (the default) issues a query without an
    ``IN (...)`` clause on ``successor_id``.  This is the retry path: any
    succession whose alert failed to post on a previous run remains
    eligible until track_history records a successful send."""
    asset_id = uuid.uuid4()
    rows = [(asset_id, "Old", "New", RelationType.REPLACE)]
    session = _make_session(rows)

    alerts = await check_succession(session)  # no new_item_ids → None default

    assert len(alerts) == 1
    assert alerts[0].user_asset_id == asset_id

    # A query was issued, and it does NOT narrow by tech_succession.successor_id.
    session.execute.assert_awaited()
    executed_stmt = session.execute.await_args.args[0]
    rendered = str(executed_stmt.compile(compile_kwargs={"literal_binds": False}))
    # Track_history dedup must still be present.
    assert "track_history" in rendered
    # No IN-list on successor_id when scanning all rows.
    assert "successor_id IN" not in rendered.replace("\n", " ")


@pytest.mark.asyncio
async def test_check_succession_none_still_filters_already_alerted():
    """Even when scanning all succession rows, the track_history NOT EXISTS
    predicate still keeps the dedup of successful sends in place."""
    session = _make_session([])  # DB filtered them all out

    alerts = await check_succession(session, None)
    assert alerts == []

    executed_stmt = session.execute.await_args.args[0]
    rendered = str(executed_stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "track_history" in rendered


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


@pytest.mark.asyncio
async def test_check_succession_dedupes_multiple_successors_for_same_asset():
    """Two ``tech_succession`` rows sharing the same Keep-ed predecessor
    (so the same ``user_asset_id``) must collapse to a single alert.  The
    SQL's track_history NOT EXISTS predicate only covers prior committed
    runs, so without in-batch dedup ``post_track_update`` would fire one
    Slack message per successor.  Representative = first encountered
    (the query is ORDER BY succession created_at ASC)."""
    asset_id = uuid.uuid4()
    rows = [
        # Earliest succession first — this should win.
        (asset_id, "Old Tech", "Newer Tech", RelationType.REPLACE),
        # Later succession for the same Keep-ed asset — must be dropped.
        (asset_id, "Old Tech", "Even Newer Tech", RelationType.ENHANCE),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session)

    assert len(alerts) == 1
    assert alerts[0].user_asset_id == asset_id
    # First-encountered representative is preserved.
    assert alerts[0].successor_title == "Newer Tech"
    assert alerts[0].relation_type is RelationType.REPLACE


@pytest.mark.asyncio
async def test_check_succession_dedupes_per_asset_across_distinct_assets():
    """Dedup is per-asset: two distinct Keep-ed assets each with multiple
    successors should yield exactly two alerts, one per asset."""
    asset_a = uuid.uuid4()
    asset_b = uuid.uuid4()
    rows = [
        (asset_a, "A old", "A new 1", RelationType.REPLACE),
        (asset_a, "A old", "A new 2", RelationType.ENHANCE),
        (asset_b, "B old", "B new 1", RelationType.FORK),
        (asset_b, "B old", "B new 2", RelationType.REPLACE),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session)

    assert [a.user_asset_id for a in alerts] == [asset_a, asset_b]
    assert [a.successor_title for a in alerts] == ["A new 1", "B new 1"]
