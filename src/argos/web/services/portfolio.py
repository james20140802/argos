"""Read-side service backing the portfolio screen (ARG-153).

``fetch_portfolio`` returns Keep assets partitioned into *active* (assets
with recent signal or any lineage row) and *quiet* groups, enriched with
per-asset ``signal_count``, ``lineage_count``, ``last_signal_at``,
``kept_since``, ``image_url``, ``category``, and ``trust_score``.

A single SQL query computes all aggregates to avoid N+1 round-trips.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import TechSuccession
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset


# ------------------------------------------------------------------ #
# Module constants
# ------------------------------------------------------------------ #

RECENT_SIGNAL_WINDOW: timedelta = timedelta(days=7)


# ------------------------------------------------------------------ #
# Public types
# ------------------------------------------------------------------ #

PortfolioCategory = Literal["Mainstream", "Alpha"]
PortfolioSort = Literal["recency", "trust"]


@dataclass(frozen=True)
class PortfolioAsset:
    id: uuid.UUID                        # user_asset.id
    tech_id: uuid.UUID
    title: str
    source_url: str
    category: Optional[CategoryType]
    image_url: Optional[str]
    trust_score: Optional[float]
    kept_since: datetime                 # user_asset.created_at (tz-aware UTC)
    last_signal_at: Optional[datetime]   # MAX(track_history.changed_at) or None
    signal_count: int                    # COUNT(track_history) within RECENT_SIGNAL_WINDOW
    lineage_count: int                   # COUNT(tech_succession involving tech_id)


@dataclass(frozen=True)
class PortfolioView:
    active: list[PortfolioAsset]   # signal_count > 0 OR lineage_count > 0
    quiet: list[PortfolioAsset]    # the rest
    category: Optional[PortfolioCategory]
    sort: PortfolioSort


# ------------------------------------------------------------------ #
# Query
# ------------------------------------------------------------------ #

async def fetch_portfolio(
    session: AsyncSession,
    *,
    category: Optional[PortfolioCategory] = None,
    sort: PortfolioSort = "recency",
) -> PortfolioView:
    """Return Keep assets partitioned into active and quiet groups.

    Aggregates (signal_count, lineage_count, last_signal_at) are computed
    in a single SQL query using correlated subqueries and aggregate functions.
    """
    if category is not None and category not in ("Mainstream", "Alpha"):
        raise ValueError(f"invalid category: {category!r}")
    if sort not in ("recency", "trust"):
        raise ValueError(f"invalid sort: {sort!r}")

    cutoff = datetime.now(timezone.utc) - RECENT_SIGNAL_WINDOW

    # ---- signal subquery: COUNT within window ----
    signal_count_sq = (
        select(func.count())
        .where(TrackHistory.user_asset_id == UserAsset.id)
        .where(TrackHistory.changed_at >= cutoff)
        .correlate(UserAsset)
        .scalar_subquery()
    )

    # ---- last_signal_at subquery: MAX(changed_at) over all time ----
    last_signal_sq = (
        select(func.max(TrackHistory.changed_at))
        .where(TrackHistory.user_asset_id == UserAsset.id)
        .correlate(UserAsset)
        .scalar_subquery()
    )

    # ---- lineage subquery: COUNT successions involving this tech_id ----
    lineage_count_sq = (
        select(func.count())
        .where(
            or_(
                TechSuccession.predecessor_id == TechItem.id,
                TechSuccession.successor_id == TechItem.id,
            )
        )
        .correlate(TechItem)
        .scalar_subquery()
    )

    # ---- sort expressions ----
    if sort == "trust":
        order_exprs = [
            TechItem.trust_score.desc().nulls_last(),
            UserAsset.created_at.desc(),
        ]
    else:  # recency
        order_exprs = [UserAsset.created_at.desc()]

    stmt = (
        select(
            UserAsset.id.label("ua_id"),
            TechItem.id.label("tech_id"),
            TechItem.title,
            TechItem.source_url,
            TechItem.category,
            TechItem.image_url,
            TechItem.trust_score,
            UserAsset.created_at.label("kept_since"),
            last_signal_sq.label("last_signal_at"),
            signal_count_sq.label("signal_count"),
            lineage_count_sq.label("lineage_count"),
        )
        .join(TechItem, UserAsset.tech_id == TechItem.id)
        .where(UserAsset.status == AssetStatus.KEEP)
        .order_by(*order_exprs)
    )

    if category is not None:
        stmt = stmt.where(TechItem.category == CategoryType(category))

    result = await session.execute(stmt)
    rows = result.all()

    assets: list[PortfolioAsset] = [
        PortfolioAsset(
            id=row.ua_id,
            tech_id=row.tech_id,
            title=row.title,
            source_url=row.source_url,
            category=row.category,
            image_url=row.image_url,
            trust_score=row.trust_score,
            kept_since=row.kept_since,
            last_signal_at=row.last_signal_at,
            signal_count=int(row.signal_count or 0),
            lineage_count=int(row.lineage_count or 0),
        )
        for row in rows
    ]

    # ---- sort in Python (mirrors SQL ORDER BY; ensures test-mock parity) ----
    if sort == "trust":

        def _sort_key(a: PortfolioAsset) -> tuple:
            # trust DESC NULLS LAST → negate, put None after real values
            trust_key = (0, -(a.trust_score or 0.0)) if a.trust_score is not None else (1, 0.0)
            kept_key = -a.kept_since.timestamp()
            return (*trust_key, kept_key)

    else:  # recency

        def _sort_key(a: PortfolioAsset) -> tuple:  # type: ignore[misc]
            return (-a.kept_since.timestamp(),)

    assets.sort(key=_sort_key)

    # ---- partition ----
    active = [a for a in assets if a.signal_count > 0 or a.lineage_count > 0]
    quiet = [a for a in assets if a.signal_count == 0 and a.lineage_count == 0]

    return PortfolioView(active=active, quiet=quiet, category=category, sort=sort)
