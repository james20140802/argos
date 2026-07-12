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
    successor_id = uuid.uuid4()
    rows = [(asset_id, "Old", "New", RelationType.REPLACE, successor_id)]
    session = _make_session(rows)

    alerts = await check_succession(session)  # no new_item_ids → None default

    assert len(alerts) == 1
    assert alerts[0].user_asset_id == asset_id
    assert alerts[0].successor_id == successor_id

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
    # Row schema: (user_asset_id, predecessor_title, successor_title, relation_type, successor_id)
    rows = [
        (
            user_asset_id,
            "Old Tech",
            "New Tech",
            RelationType.REPLACE,
            successor_id,
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
    assert alert.successor_id == successor_id


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
    # ARG-204: dedup predicate must correlate on successor_id (via
    # changed_from), not just changed_to — this is what makes the dedup
    # per-pair instead of per-asset.
    assert "changed_from" in rendered


@pytest.mark.asyncio
async def test_check_succession_multiple_alerts_preserves_order():
    successor_a = uuid.uuid4()
    successor_b = uuid.uuid4()
    asset_a = uuid.uuid4()
    asset_b = uuid.uuid4()
    rows = [
        (asset_a, "A old", "A new", RelationType.ENHANCE, successor_a),
        (asset_b, "B old", "B new", RelationType.FORK, successor_b),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session, [successor_a, successor_b])

    assert [a.user_asset_id for a in alerts] == [asset_a, asset_b]
    assert [a.relation_type for a in alerts] == [
        RelationType.ENHANCE,
        RelationType.FORK,
    ]
    assert [a.successor_id for a in alerts] == [successor_a, successor_b]


@pytest.mark.asyncio
async def test_check_succession_returns_alert_per_successor_for_same_asset():
    """ARG-204 fix: dedup granularity is per (user_asset, successor) pair,
    not per user_asset.  A Keep-ed asset with two *different* successors
    must yield two alerts — one per successor — so a second, distinct
    succession is never silently suppressed just because the asset already
    received an earlier, different succession alert."""
    asset_id = uuid.uuid4()
    successor_1 = uuid.uuid4()
    successor_2 = uuid.uuid4()
    rows = [
        (asset_id, "Old Tech", "Newer Tech", RelationType.REPLACE, successor_1),
        (asset_id, "Old Tech", "Even Newer Tech", RelationType.ENHANCE, successor_2),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session)

    assert len(alerts) == 2
    assert {a.successor_id for a in alerts} == {successor_1, successor_2}
    assert all(a.user_asset_id == asset_id for a in alerts)


@pytest.mark.asyncio
async def test_check_succession_dedupes_duplicate_pair_rows():
    """If the exact same (asset, successor) pair appears twice in the raw
    result set (e.g. duplicate/overlapping tech_succession rows), in-batch
    dedup still collapses it to a single alert — first encountered wins.
    This is the (asset, successor)-pair analog of the old asset-only
    in-batch dedup."""
    asset_id = uuid.uuid4()
    successor_id = uuid.uuid4()
    rows = [
        # Earliest row for this exact pair — this should win.
        (asset_id, "Old Tech", "Newer Tech", RelationType.REPLACE, successor_id),
        # Duplicate row for the *same* pair — must be dropped.
        (asset_id, "Old Tech", "Newer Tech (dup)", RelationType.ENHANCE, successor_id),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session)

    assert len(alerts) == 1
    assert alerts[0].user_asset_id == asset_id
    assert alerts[0].successor_id == successor_id
    # First-encountered representative is preserved.
    assert alerts[0].successor_title == "Newer Tech"
    assert alerts[0].relation_type is RelationType.REPLACE


@pytest.mark.asyncio
async def test_check_succession_returns_all_pairs_across_distinct_assets():
    """Two distinct Keep-ed assets, each with two distinct successors,
    yield four alerts total — one per (user_asset, successor) pair."""
    asset_a = uuid.uuid4()
    asset_b = uuid.uuid4()
    succ_a1, succ_a2 = uuid.uuid4(), uuid.uuid4()
    succ_b1, succ_b2 = uuid.uuid4(), uuid.uuid4()
    rows = [
        (asset_a, "A old", "A new 1", RelationType.REPLACE, succ_a1),
        (asset_a, "A old", "A new 2", RelationType.ENHANCE, succ_a2),
        (asset_b, "B old", "B new 1", RelationType.FORK, succ_b1),
        (asset_b, "B old", "B new 2", RelationType.REPLACE, succ_b2),
    ]
    session = _make_session(rows)

    alerts = await check_succession(session)

    assert len(alerts) == 4
    assert [a.user_asset_id for a in alerts] == [asset_a, asset_a, asset_b, asset_b]
    assert [a.successor_id for a in alerts] == [succ_a1, succ_a2, succ_b1, succ_b2]
