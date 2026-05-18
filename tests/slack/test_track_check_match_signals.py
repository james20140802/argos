"""ARG-115: unit tests for match_signals pgvector cosine query.

Tests use mocked AsyncSession.execute so no real DB is required.
Covers:
- Explicit empty list short-circuits to [] without issuing a query.
- new_item_ids=None issues query without an IN-list.
- Only (user_asset_id, new_item_id) pairs not in track_history are returned.
- Threshold boundary: >=0.85 rows pass, <0.85 rows are excluded by DB.
- SignalMatch dataclass fields are populated correctly.
- Multiple matches from different assets are returned.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.slack.services.track_check import (
    SIGNAL_MATCHED,
    SIGNAL_SIMILARITY_THRESHOLD,
    SignalMatch,
    match_signals,
)


def _make_row(
    *,
    user_asset_id: uuid.UUID | None = None,
    keep_item_id: uuid.UUID | None = None,
    keep_item_title: str = "Keep Tech",
    new_item_id: uuid.UUID | None = None,
    new_item_title: str = "New Tech",
    new_item_url: str = "https://example.com/new",
    similarity_score: float = 0.92,
):
    """Build a minimal row-like object matching the SQL result columns."""
    row = MagicMock()
    row.user_asset_id = user_asset_id or uuid.uuid4()
    row.keep_item_id = keep_item_id or uuid.uuid4()
    row.keep_item_title = keep_item_title
    row.new_item_id = new_item_id or uuid.uuid4()
    row.new_item_title = new_item_title
    row.new_item_url = new_item_url
    row.similarity_score = similarity_score
    return row


def _make_session(rows: list) -> AsyncMock:
    """Mock an AsyncSession.execute that returns fetchall() = rows."""
    result = MagicMock()
    result.fetchall.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    return session


# ---------------------------------------------------------------------------
# Short-circuit / no-query cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_signals_explicit_empty_list_returns_empty_without_query():
    """An explicit [] means 'look at nothing' — short-circuit, no DB hit."""
    session = AsyncMock()
    matches = await match_signals(session, [])
    assert matches == []
    session.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# Query content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_signals_none_issues_query_without_in_list():
    """new_item_ids=None (default) issues a query but without an IN-list filter."""
    session = _make_session([])
    await match_signals(session, None)

    session.execute.assert_awaited_once()
    executed_sql = str(session.execute.await_args.args[0])
    # Should NOT narrow by ni.id IN (...)
    assert "ni.id IN" not in executed_sql
    # Must reference the threshold sentinel
    assert "signal_matched" in str(session.execute.await_args.args[1]).lower() or \
           "signal_matched" in executed_sql or \
           SIGNAL_MATCHED in str(session.execute.await_args.args[1])


@pytest.mark.asyncio
async def test_match_signals_with_new_item_ids_adds_in_filter():
    """Passing a non-empty list adds an IN (...) clause on ni.id."""
    nid = uuid.uuid4()
    session = _make_session([])
    await match_signals(session, [nid])

    session.execute.assert_awaited_once()
    executed_sql = str(session.execute.await_args.args[0])
    assert "ni.id IN" in executed_sql


# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_signals_returns_signal_match_objects():
    """Rows returned by the DB are correctly mapped to SignalMatch dataclasses."""
    asset_id = uuid.uuid4()
    keep_id = uuid.uuid4()
    new_id = uuid.uuid4()
    row = _make_row(
        user_asset_id=asset_id,
        keep_item_id=keep_id,
        keep_item_title="Keep Tech",
        new_item_id=new_id,
        new_item_title="New Tech",
        new_item_url="https://example.com/new",
        similarity_score=0.92,
    )
    session = _make_session([row])

    matches = await match_signals(session)

    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, SignalMatch)
    assert m.user_asset_id == asset_id
    assert m.keep_item_id == keep_id
    assert m.keep_item_title == "Keep Tech"
    assert m.new_item_id == new_id
    assert m.new_item_title == "New Tech"
    assert m.new_item_url == "https://example.com/new"
    assert abs(m.similarity_score - 0.92) < 1e-9


@pytest.mark.asyncio
async def test_match_signals_empty_rows_returns_empty_list():
    """No matching rows → empty list."""
    session = _make_session([])
    matches = await match_signals(session)
    assert matches == []


@pytest.mark.asyncio
async def test_match_signals_multiple_matches():
    """Multiple rows from the DB are returned as multiple SignalMatch objects."""
    rows = [_make_row(similarity_score=0.91), _make_row(similarity_score=0.88)]
    session = _make_session(rows)

    matches = await match_signals(session)
    assert len(matches) == 2
    assert all(isinstance(m, SignalMatch) for m in matches)


# ---------------------------------------------------------------------------
# Threshold / sentinel semantics (DB-level filtering verified via SQL rendering)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_signals_threshold_in_query_params():
    """The threshold value must be passed as a query parameter."""
    session = _make_session([])
    await match_signals(session)

    params = session.execute.await_args.args[1]
    assert "threshold" in params
    assert params["threshold"] == SIGNAL_SIMILARITY_THRESHOLD


@pytest.mark.asyncio
async def test_match_signals_sentinel_in_query_params():
    """The SIGNAL_MATCHED sentinel must be passed as a query parameter."""
    session = _make_session([])
    await match_signals(session)

    params = session.execute.await_args.args[1]
    assert "sentinel" in params
    assert params["sentinel"] == SIGNAL_MATCHED


@pytest.mark.asyncio
async def test_match_signals_not_exists_dedup_in_sql():
    """The rendered SQL must include a NOT EXISTS clause referencing track_history."""
    session = _make_session([])
    await match_signals(session)

    executed_sql = str(session.execute.await_args.args[0])
    assert "NOT EXISTS" in executed_sql
    assert "track_history" in executed_sql


@pytest.mark.asyncio
async def test_match_signals_dedup_uses_changed_from_as_new_item_id():
    """Dedup is per (user_asset_id, new_item_id): the SQL must reference
    changed_from to encode the new_item_id, not just changed_to."""
    session = _make_session([])
    await match_signals(session)

    executed_sql = str(session.execute.await_args.args[0])
    assert "changed_from" in executed_sql
    assert "changed_to" in executed_sql


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------


def test_signal_match_is_frozen():
    """SignalMatch must be frozen (immutable) like SuccessionAlert."""
    m = SignalMatch(
        user_asset_id=uuid.uuid4(),
        keep_item_id=uuid.uuid4(),
        keep_item_title="A",
        new_item_id=uuid.uuid4(),
        new_item_title="B",
        new_item_url="https://example.com",
        similarity_score=0.9,
    )
    with pytest.raises((AttributeError, TypeError)):
        m.similarity_score = 0.5  # type: ignore[misc]
