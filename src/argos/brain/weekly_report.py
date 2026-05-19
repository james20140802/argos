"""Weekly Keep portfolio aggregation (ARG-122).

Pure DB-layer service: callable by ARG-123's Slack renderer and by future
non-Slack consumers (CLI, report exports). No Slack imports, no commits.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import TechItem
from argos.models.tech_succession import TechSuccession
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset

WEEKLY_WINDOW = timedelta(days=7)


@dataclass(frozen=True)
class WeeklyKeepItem:
    tech_id: uuid.UUID
    title: str
    signals_7d: int
    successions_7d: int
    last_monitored_at: datetime | None


@dataclass(frozen=True)
class WeeklyKeepReport:
    total_keep_count: int
    items: list[WeeklyKeepItem]
    window_start: datetime
    window_end: datetime


async def build_weekly_keep_report(
    session: AsyncSession,
    *,
    now_utc: datetime | None = None,
) -> WeeklyKeepReport:
    """Aggregate 7-day signal/succession activity for every Keep-ed asset.

    Parameters
    ----------
    session:
        Async SQLAlchemy session. Not committed by this function.
    now_utc:
        Override the window's upper bound. Defaults to
        ``datetime.now(timezone.utc)``. The window is ``[now_utc - 7d, now_utc)``.
    """
    window_end = now_utc if now_utc is not None else datetime.now(timezone.utc)
    window_start = window_end - WEEKLY_WINDOW

    # Q1: Keep portfolio base set (one row per Keep-ed user_asset).
    keep_stmt = (
        select(
            UserAsset.id.label("user_asset_id"),
            UserAsset.tech_id.label("tech_id"),
            UserAsset.last_monitored_at.label("last_monitored_at"),
            TechItem.title.label("title"),
        )
        .join(TechItem, TechItem.id == UserAsset.tech_id)
        .where(UserAsset.status == AssetStatus.KEEP)
        .order_by(TechItem.title.asc())
    )
    keep_rows = (await session.execute(keep_stmt)).all()

    if not keep_rows:
        return WeeklyKeepReport(
            total_keep_count=0,
            items=[],
            window_start=window_start,
            window_end=window_end,
        )

    # Q2: signals_7d - track_history rows in window, grouped by user_asset_id.
    signal_stmt = (
        select(
            TrackHistory.user_asset_id.label("user_asset_id"),
            func.count(TrackHistory.id).label("count"),
        )
        .where(TrackHistory.changed_at >= window_start)
        .where(TrackHistory.changed_at < window_end)
        .group_by(TrackHistory.user_asset_id)
    )
    signal_rows = (await session.execute(signal_stmt)).all()
    signals_by_asset: dict[uuid.UUID, int] = {row[0]: row[1] for row in signal_rows}

    # Q3: successions_7d - tech_succession rows in window, grouped by predecessor_id.
    succession_stmt = (
        select(
            TechSuccession.predecessor_id.label("tech_id"),
            func.count(TechSuccession.id).label("count"),
        )
        .where(TechSuccession.created_at >= window_start)
        .where(TechSuccession.created_at < window_end)
        .group_by(TechSuccession.predecessor_id)
    )
    succession_rows = (await session.execute(succession_stmt)).all()
    successions_by_tech: dict[uuid.UUID, int] = {
        row[0]: row[1] for row in succession_rows
    }

    items: list[WeeklyKeepItem] = []
    for row in keep_rows:
        # Real Rows expose .user_asset_id / .tech_id / .last_monitored_at / .title;
        # the mocked tests deliver plain tuples in the same column order.
        if hasattr(row, "tech_id"):
            user_asset_id = row.user_asset_id
            tech_id = row.tech_id
            last_monitored_at = row.last_monitored_at
            title = row.title
        else:
            user_asset_id, tech_id, last_monitored_at, title = row
        items.append(
            WeeklyKeepItem(
                tech_id=tech_id,
                title=title,
                signals_7d=signals_by_asset.get(user_asset_id, 0),
                successions_7d=successions_by_tech.get(tech_id, 0),
                last_monitored_at=last_monitored_at,
            )
        )

    return WeeklyKeepReport(
        total_keep_count=len(items),
        items=items,
        window_start=window_start,
        window_end=window_end,
    )
