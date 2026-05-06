from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
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

    - 자산이 없으면 target_status로 생성한다 (CREATED).
    - 같은 상태면 아무 것도 하지 않는다 (NOOP).
    - 다른 상태면 TrackHistory 행을 추가하고 상태/last_monitored_at을 갱신한다 (TRANSITIONED).
    """
    result = await session.execute(
        select(UserAsset).where(UserAsset.tech_id == tech_id)
    )
    asset = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if asset is None:
        session.add(
            UserAsset(
                tech_id=tech_id,
                status=target_status,
                last_monitored_at=now,
            )
        )
        return TransitionOutcome.CREATED

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
