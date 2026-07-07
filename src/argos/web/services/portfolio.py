"""Read-side service backing the portfolio screen (ARG-153).

``fetch_portfolio`` returns Keep assets partitioned into *active* (assets
with recent signal or any lineage row) and *quiet* groups, enriched with
per-asset ``signal_count``, ``lineage_count``, ``last_signal_at``,
``kept_since``, ``image_url``, ``category``, and ``trust_score``.

A single SQL query computes all aggregates to avoid N+1 round-trips.
"""
from __future__ import annotations

import base64
import json
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

PAGE_SIZE: int = 20


def encode_portfolio_cursor(
    kept_since: datetime, ua_id: uuid.UUID, trust_score: Optional[float]
) -> str:
    """Opaque keyset cursor for the portfolio sort position.

    Carries the full compound key so both sort modes can page from it:
    ``kept_since`` + ``ua_id`` (recency, and the tie-breaker for trust) plus
    ``trust_score`` (the trust-sort primary key; may be ``None``).
    """
    if kept_since.tzinfo is None:
        kept_since = kept_since.replace(tzinfo=timezone.utc)
    payload = {
        "k": kept_since.astimezone(timezone.utc).isoformat(),
        "u": ua_id.hex,
        "t": trust_score,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_portfolio_cursor(token: str) -> tuple[datetime, uuid.UUID, Optional[float]]:
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        kept = datetime.fromisoformat(payload["k"])
        if kept.tzinfo is None:
            kept = kept.replace(tzinfo=timezone.utc)
        ua_id = uuid.UUID(payload["u"])
        trust = payload["t"]
        if trust is not None:
            trust = float(trust)
        return kept, ua_id, trust
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid portfolio cursor: {token!r}") from exc


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
    next_cursor: Optional[str] = None


# ------------------------------------------------------------------ #
# Query
# ------------------------------------------------------------------ #

async def fetch_portfolio(
    session: AsyncSession,
    *,
    category: Optional[PortfolioCategory] = None,
    sort: PortfolioSort = "recency",
    cursor: Optional[str] = None,
    limit: int = PAGE_SIZE,
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

    # ``track_history`` is a shared log: ``transition_asset`` writes ordinary
    # status changes (changed_to ∈ AssetStatus values) while real signal alerts
    # are recorded with these sentinel ``changed_to`` values.  Portfolio signal
    # aggregates must count only the latter, otherwise an ordinary status flip
    # (e.g. Archive→Keep) would falsely surface as a "new signal" and move the
    # asset into the active group.  Imported lazily so app construction does not
    # pull ``argos.database`` into the import graph (release CI has no Postgres).
    from argos.slack.services.track_check import SIGNAL_MATCHED, SUCCESSION_ALERTED

    signal_sentinels = (SUCCESSION_ALERTED, SIGNAL_MATCHED)

    # ---- signal subquery: COUNT signal alerts within window ----
    signal_count_sq = (
        select(func.count())
        .where(TrackHistory.user_asset_id == UserAsset.id)
        .where(TrackHistory.changed_to.in_(signal_sentinels))
        .where(TrackHistory.changed_at >= cutoff)
        .correlate(UserAsset)
        .scalar_subquery()
    )

    # ---- last_signal_at subquery: MAX(changed_at) over signal alerts ----
    last_signal_sq = (
        select(func.max(TrackHistory.changed_at))
        .where(TrackHistory.user_asset_id == UserAsset.id)
        .where(TrackHistory.changed_to.in_(signal_sentinels))
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
            UserAsset.id.desc(),
        ]
    else:  # recency
        order_exprs = [UserAsset.created_at.desc(), UserAsset.id.desc()]

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

    if cursor is not None:
        cur_kept, cur_ua, cur_trust = decode_portfolio_cursor(cursor)
        kept_tie = (UserAsset.created_at < cur_kept) | (
            (UserAsset.created_at == cur_kept) & (UserAsset.id < cur_ua)
        )
        if sort == "trust":
            if cur_trust is None:
                # Cursor is in the NULLS-LAST tail: only other null-trust rows
                # can sort after it.
                stmt = stmt.where(TechItem.trust_score.is_(None) & kept_tie)
            else:
                stmt = stmt.where(
                    TechItem.trust_score.is_(None)
                    | (TechItem.trust_score < cur_trust)
                    | ((TechItem.trust_score == cur_trust) & kept_tie)
                )
        else:  # recency
            stmt = stmt.where(kept_tie)

    stmt = stmt.limit(limit + 1)

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
    # The trailing ``-a.id.int`` element is not just test scaffolding: it is
    # load-bearing for `next_cursor` correctness in production. This re-sort
    # decides `page[-1]`, which seeds the next cursor — it must match the SQL
    # tiebreak (``UserAsset.id.desc()``) exactly, rather than relying on
    # Python's stable sort to implicitly preserve the DB's row order.
    if sort == "trust":

        def _sort_key(a: PortfolioAsset) -> tuple:
            # trust DESC NULLS LAST → negate, put None after real values
            trust_key = (0, -a.trust_score) if a.trust_score is not None else (1, 0.0)
            kept_key = -a.kept_since.timestamp()
            return (*trust_key, kept_key, -a.id.int)

    else:  # recency

        def _sort_key(a: PortfolioAsset) -> tuple:  # type: ignore[misc]
            return (-a.kept_since.timestamp(), -a.id.int)

    assets.sort(key=_sort_key)

    # ---- trim to one page; the (limit+1)-th row only signals "more" ----
    has_more = len(assets) > limit
    page = assets[:limit]
    next_cursor = (
        encode_portfolio_cursor(page[-1].kept_since, page[-1].id, page[-1].trust_score)
        if has_more and page
        else None
    )

    # ---- partition the loaded page (active/quiet split within this page) ----
    active = [a for a in page if a.signal_count > 0 or a.lineage_count > 0]
    quiet = [a for a in page if a.signal_count == 0 and a.lineage_count == 0]

    return PortfolioView(
        active=active, quiet=quiet, category=category, sort=sort, next_cursor=next_cursor
    )
