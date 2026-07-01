from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset


class TransitionOutcome(str, enum.Enum):
    CREATED = "created"
    TRANSITIONED = "transitioned"
    NOOP = "noop"


async def transition_asset(
    session: AsyncSession,
    tech_id: uuid.UUID,
    target_status: AssetStatus,
) -> TransitionOutcome:
    """tech_id의 UserAsset을 target_status로 전이하고 이력을 기록한다.

    동시 클릭으로 인한 중복 row 생성을 막기 위해 ``INSERT ... ON CONFLICT DO
    NOTHING`` 으로 자산을 atomic하게 만들고, 이미 존재하는 경우 ``SELECT ...
    FOR UPDATE`` 로 행을 잠가 상태 전이를 직렬화한다.

    - 자산이 없으면 target_status로 생성한다 (CREATED).
    - 같은 상태면 아무 것도 하지 않는다 (NOOP).
    - 다른 상태면 TrackHistory 행을 추가하고 상태/last_monitored_at을 갱신한다 (TRANSITIONED).
    """
    now = datetime.now(timezone.utc)

    insert_stmt = (
        pg_insert(UserAsset)
        .values(
            tech_id=tech_id,
            status=target_status,
            last_monitored_at=now,
        )
        .on_conflict_do_nothing(index_elements=["tech_id"])
        .returning(UserAsset.id)
    )
    insert_result = await session.execute(insert_stmt)
    if insert_result.scalar_one_or_none() is not None:
        return TransitionOutcome.CREATED

    locked = await session.execute(
        select(UserAsset)
        .where(UserAsset.tech_id == tech_id)
        .with_for_update()
    )
    asset = locked.scalar_one()

    if asset.status == target_status:
        return TransitionOutcome.NOOP

    previous = asset.status
    asset.status = target_status
    asset.last_monitored_at = now
    session.add(
        TrackHistory(
            user_asset_id=asset.id,
            changed_from=previous.value,
            changed_to=target_status.value,
        )
    )
    return TransitionOutcome.TRANSITIONED


class ToggleOutcome(str, enum.Enum):
    SET = "set"          # created, or switched into target_status
    REMOVED = "removed"  # toggled off — the triage decision was cleared


async def toggle_asset(
    session: AsyncSession,
    tech_id: uuid.UUID,
    target_status: AssetStatus,
) -> ToggleOutcome:
    """Toggle a feed triage decision on ``tech_id``.

    - No asset, or an asset in a *different* status → set ``target_status``
      (delegates to :func:`transition_asset`, which creates or switches and logs
      the transition). Returns ``SET``.
    - Asset already in ``target_status`` → clear the decision by deleting the
      UserAsset, returning the item to untriaged (its ``track_history`` rows
      cascade). Returns ``REMOVED``.

    Unlike ``transition_asset`` this never NOOPs: pressing the already-active
    button is a deliberate un-toggle.
    """
    existing = (
        await session.execute(
            select(UserAsset).where(UserAsset.tech_id == tech_id)
        )
    ).scalar_one_or_none()
    if existing is not None and existing.status == target_status:
        await session.delete(existing)
        return ToggleOutcome.REMOVED
    await transition_asset(session, tech_id, target_status)
    return ToggleOutcome.SET
