"""Read-side service for the 관측 신호 ticker (feed right rail).

``fetch_activity`` returns the most recent *signal* rows from the shared
``track_history`` log — the same alert rows the portfolio counts to mark an
asset active (``signal_matched`` / ``succession_alerted``), but scoped GLOBALLY
rather than to one item. This is the live "what the observatory just picked up"
stream that sits beside the discovery feed (a distinct stream, not a copy of the
newest-items feed).

Scoped to assets that are *currently* ``Keep`` — mirroring the portfolio signal
counts (``portfolio.py`` filters ``UserAsset.status == KEEP``). An asset that was
Kept (accruing signal history) and later Passed/Archived keeps its old signal
rows, but those must not keep surfacing as live 관측 신호 once tracking stopped.

Ordinary status transitions (Keep / Pass / Archive) are intentionally excluded:
the ticker is about *signals the system surfaced*, not the user's own actions.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import String, and_, cast, select
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import TechItem
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset


# How many signal rows the ticker shows. The rail is a glanceable digest, not a
# full history — the detail page already carries the per-item timeline.
ACTIVITY_LIMIT: int = 12


@dataclass(frozen=True)
class ActivityEntry:
    """One signal row for the ticker.

    * ``kind == "signal"`` — a new tech_item matched a Keep asset. ``matched_*``
      describes that new item (the trigger); both ``None`` if it was deleted.
    * ``kind == "succession"`` — a succession/replacement alert for the asset;
      it carries no specific matched item.
    """

    kind: str  # "signal" | "succession"
    tech_id: uuid.UUID          # the Keep asset that signalled (link target)
    tech_title: str
    changed_at: datetime
    matched_tech_id: Optional[uuid.UUID] = None
    matched_title: Optional[str] = None


async def fetch_activity(
    session: AsyncSession, limit: int = ACTIVITY_LIMIT
) -> list[ActivityEntry]:
    """Return the most recent global signal-alert rows, newest first."""
    # Lazy import keeps the web layer decoupled from the slack module (mirrors
    # detail.py / portfolio.py): importing track_check at module scope pulls
    # argos.database into the import graph, and release CI has no Postgres.
    from argos.slack.services.track_check import SIGNAL_MATCHED, SUCCESSION_ALERTED

    Matched = aliased(TechItem)
    stmt = (
        select(
            TrackHistory.changed_to,
            TrackHistory.changed_at,
            TechItem.id.label("tech_id"),
            TechItem.title.label("tech_title"),
            Matched.id.label("matched_id"),
            Matched.title.label("matched_title"),
        )
        .join(UserAsset, UserAsset.id == TrackHistory.user_asset_id)
        .join(TechItem, TechItem.id == UserAsset.tech_id)
        .join(
            Matched,
            and_(
                TrackHistory.changed_to == SIGNAL_MATCHED,
                cast(Matched.id, String) == TrackHistory.changed_from,
            ),
            isouter=True,
        )
        .where(
            UserAsset.status == AssetStatus.KEEP,
            TrackHistory.changed_to.in_((SIGNAL_MATCHED, SUCCESSION_ALERTED)),
        )
        .order_by(TrackHistory.changed_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        ActivityEntry(
            kind="signal" if row.changed_to == SIGNAL_MATCHED else "succession",
            tech_id=row.tech_id,
            tech_title=row.tech_title,
            changed_at=row.changed_at,
            matched_tech_id=row.matched_id,
            matched_title=row.matched_title,
        )
        for row in rows
    ]
