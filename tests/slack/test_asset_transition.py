from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset
from argos.slack.services.asset_transition import (
    TransitionOutcome,
    transition_asset,
)


def _mock_session(existing_asset: UserAsset | None) -> tuple[AsyncMock, list]:
    added: list = []
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing_asset
    session.execute = AsyncMock(return_value=result)
    session.add = lambda obj: added.append(obj)
    return session, added


@pytest.mark.asyncio
async def test_transition_creates_new_asset_when_missing(tech_id):
    session, added = _mock_session(None)

    outcome = await transition_asset(session, tech_id, AssetStatus.KEEP)

    assert outcome is TransitionOutcome.CREATED
    assert len(added) == 1
    assert isinstance(added[0], UserAsset)
    assert added[0].status is AssetStatus.KEEP
    assert added[0].tech_id == tech_id
    assert added[0].last_monitored_at is not None


@pytest.mark.asyncio
async def test_transition_noop_when_status_unchanged(tech_id):
    existing = UserAsset(
        id=uuid.uuid4(),
        tech_id=tech_id,
        status=AssetStatus.KEEP,
    )
    session, added = _mock_session(existing)

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
    session, added = _mock_session(existing)

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
    session, added = _mock_session(existing)

    outcome = await transition_asset(session, tech_id, AssetStatus.KEEP)

    assert outcome is TransitionOutcome.TRANSITIONED
    history_rows = [obj for obj in added if isinstance(obj, TrackHistory)]
    assert len(history_rows) == 1
    assert history_rows[0].changed_from == AssetStatus.ARCHIVED.value
    assert history_rows[0].changed_to == AssetStatus.KEEP.value
