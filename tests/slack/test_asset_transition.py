from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset
from argos.slack.services.asset_transition import (
    ToggleOutcome,
    TransitionOutcome,
    toggle_asset,
    transition_asset,
)


def _mock_session(
    *,
    inserted_id: uuid.UUID | None,
    existing_asset: UserAsset | None = None,
) -> tuple[AsyncMock, list]:
    """Mock an AsyncSession exercising the upsert + lock-and-read flow.

    `inserted_id` is what the ON CONFLICT INSERT's RETURNING yields:
    - a UUID when the row was newly inserted (CREATED path)
    - None when the row already existed (NOOP/TRANSITIONED path)

    `existing_asset` is what the follow-up SELECT ... FOR UPDATE returns when
    the insert was a no-op.
    """
    added: list = []
    insert_result = MagicMock()
    insert_result.scalar_one_or_none.return_value = inserted_id
    select_result = MagicMock()
    select_result.scalar_one.return_value = existing_asset
    select_result.scalar_one_or_none.return_value = existing_asset

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[insert_result, select_result])
    session.add = lambda obj: added.append(obj)
    return session, added


@pytest.mark.asyncio
async def test_transition_creates_new_asset_when_missing(tech_id):
    new_id = uuid.uuid4()
    session, added = _mock_session(inserted_id=new_id)

    outcome = await transition_asset(session, tech_id, AssetStatus.KEEP)

    assert outcome is TransitionOutcome.CREATED
    assert added == []
    assert session.execute.call_count == 1


@pytest.mark.asyncio
async def test_transition_noop_when_status_unchanged(tech_id):
    existing = UserAsset(
        id=uuid.uuid4(),
        tech_id=tech_id,
        status=AssetStatus.KEEP,
    )
    session, added = _mock_session(inserted_id=None, existing_asset=existing)

    outcome = await transition_asset(session, tech_id, AssetStatus.KEEP)

    assert outcome is TransitionOutcome.NOOP
    assert added == []
    assert existing.status is AssetStatus.KEEP


@pytest.mark.asyncio
async def test_transition_logs_history_on_status_change(tech_id):
    asset_id = uuid.uuid4()
    existing = UserAsset(
        id=asset_id,
        tech_id=tech_id,
        status=AssetStatus.KEEP,
    )
    session, added = _mock_session(inserted_id=None, existing_asset=existing)

    outcome = await transition_asset(session, tech_id, AssetStatus.ARCHIVED)

    assert outcome is TransitionOutcome.TRANSITIONED
    assert existing.status is AssetStatus.ARCHIVED
    assert existing.last_monitored_at is not None
    history_rows = [obj for obj in added if isinstance(obj, TrackHistory)]
    assert len(history_rows) == 1
    history = history_rows[0]
    assert history.user_asset_id == asset_id
    assert history.changed_from == AssetStatus.KEEP.value
    assert history.changed_to == AssetStatus.ARCHIVED.value


@pytest.mark.asyncio
async def test_transition_archived_to_keep_logs_history(tech_id):
    existing = UserAsset(
        id=uuid.uuid4(),
        tech_id=tech_id,
        status=AssetStatus.ARCHIVED,
    )
    session, added = _mock_session(inserted_id=None, existing_asset=existing)

    outcome = await transition_asset(session, tech_id, AssetStatus.KEEP)

    assert outcome is TransitionOutcome.TRANSITIONED
    history_rows = [obj for obj in added if isinstance(obj, TrackHistory)]
    assert len(history_rows) == 1
    assert history_rows[0].changed_from == AssetStatus.ARCHIVED.value
    assert history_rows[0].changed_to == AssetStatus.KEEP.value


# --- toggle_asset (feed triage toggle) ---------------------------------------


@pytest.mark.asyncio
async def test_toggle_clear_deletes_when_active_and_status_matches(tech_id):
    """currently_active=True on a row still in the target status clears it."""
    existing = UserAsset(id=uuid.uuid4(), tech_id=tech_id, status=AssetStatus.KEEP)
    sel = MagicMock()
    sel.scalar_one_or_none.return_value = existing
    session = AsyncMock()
    session.execute = AsyncMock(return_value=sel)
    session.delete = AsyncMock()

    outcome = await toggle_asset(
        session, tech_id, AssetStatus.KEEP, currently_active=True
    )

    assert outcome is ToggleOutcome.REMOVED
    session.delete.assert_awaited_once_with(existing)


@pytest.mark.asyncio
async def test_toggle_clear_is_noop_when_row_already_gone(tech_id):
    """Stale active card: the client rendered ✓ but the DB row is already gone.
    Clearing must not error or recreate — REMOVED with no delete/insert."""
    sel = MagicMock()
    sel.scalar_one_or_none.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=sel)
    session.delete = AsyncMock()

    outcome = await toggle_asset(
        session, tech_id, AssetStatus.KEEP, currently_active=True
    )

    assert outcome is ToggleOutcome.REMOVED
    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_toggle_clear_leaves_different_status_untouched(tech_id):
    """A stale ✓Keep press while the DB actually holds Archived must not wipe
    the Archived decision the user never acted on (guarded delete)."""
    existing = UserAsset(id=uuid.uuid4(), tech_id=tech_id, status=AssetStatus.ARCHIVED)
    sel = MagicMock()
    sel.scalar_one_or_none.return_value = existing
    session = AsyncMock()
    session.execute = AsyncMock(return_value=sel)
    session.delete = AsyncMock()

    outcome = await toggle_asset(
        session, tech_id, AssetStatus.KEEP, currently_active=True
    )

    assert outcome is ToggleOutcome.REMOVED
    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_toggle_set_creates_when_inactive(tech_id):
    """currently_active=False → set (delegates straight to transition_asset)."""
    insert_res = MagicMock()
    insert_res.scalar_one_or_none.return_value = uuid.uuid4()  # RETURNING id → CREATED
    session = AsyncMock()
    session.execute = AsyncMock(return_value=insert_res)
    session.delete = AsyncMock()

    outcome = await toggle_asset(
        session, tech_id, AssetStatus.KEEP, currently_active=False
    )

    assert outcome is ToggleOutcome.SET
    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_toggle_set_switches_when_inactive_and_different_status(tech_id):
    """Pressing an inactive Keep while the DB holds Archived switches to Keep
    and logs the transition (not a delete)."""
    existing = UserAsset(id=uuid.uuid4(), tech_id=tech_id, status=AssetStatus.ARCHIVED)
    insert_res = MagicMock()
    insert_res.scalar_one_or_none.return_value = None  # exists → transition path
    lock_res = MagicMock()
    lock_res.scalar_one.return_value = existing
    added: list = []
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[insert_res, lock_res])
    session.add = lambda obj: added.append(obj)
    session.delete = AsyncMock()

    outcome = await toggle_asset(
        session, tech_id, AssetStatus.KEEP, currently_active=False
    )

    assert outcome is ToggleOutcome.SET
    assert existing.status is AssetStatus.KEEP
    session.delete.assert_not_called()
    assert any(isinstance(obj, TrackHistory) for obj in added)
