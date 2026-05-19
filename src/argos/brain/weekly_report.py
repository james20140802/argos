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
    raise NotImplementedError  # filled in Task 2/3
