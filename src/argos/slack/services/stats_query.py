"""Stats query service for `argos stats` subcommand (ARG-66).

All SQL aggregation lives here; the CLI handler is a thin async wrapper.
Pure helpers (classify_source, safe_pct) have no DB dependency and can be
unit-tested directly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import TechItem
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def classify_source(url: str) -> str:
    """Map a source URL to a display label for collection stats.

    Mapping:
      github.com / githubusercontent.com / gist.github.com → GitHub
      news.ycombinator.com / hacker-news.firebaseio.com   → HN
      arxiv.org                                            → arXiv
      everything else                                      → RSS
        (called "RSS" rather than "Other" because the majority of unrecognised
         sources in practice come from RSS/Atom feeds; a one-label bucket is
         simpler and more readable in the output)
    """
    if not url:
        return "RSS"
    try:
        host = urlsplit(url).hostname or ""
    except Exception:
        return "RSS"

    host = host.lower()

    # GitHub family (github.com + all subdomains, githubusercontent.com)
    if host == "github.com" or host.endswith(".github.com") or host.endswith(".githubusercontent.com"):
        return "GitHub"

    # Hacker News family
    if host == "news.ycombinator.com" or host == "hacker-news.firebaseio.com":
        return "HN"

    # arXiv
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        return "arXiv"

    return "RSS"


def safe_pct(numerator: int, denominator: int) -> int:
    """Return integer percentage (0-100), guarding against zero division.

    Returns 0 when denominator is 0 (shown as "0%" in output).
    """
    if denominator == 0:
        return 0
    return round(numerator * 100 / denominator)


# ---------------------------------------------------------------------------
# DB query
# ---------------------------------------------------------------------------


async def fetch_stats_summary(
    session: AsyncSession,
    *,
    days: int,
) -> dict:
    """Aggregate all stats needed for `argos stats` output.

    Returns a dict with keys:
      total_items, github_count, hn_count, rss_count, arxiv_count,
      valid_count, new_saved_count, keep_count, pass_count, unclassified_count,
      total_keep_cumulative, track_alert_count
    """
    now = datetime.now(tz=timezone.utc)
    since = now - timedelta(days=days)

    # ── Collection: all tech_items created in the window ──────────────────
    items_in_window_result = await session.execute(
        select(TechItem.source_url).where(TechItem.created_at >= since)
    )
    urls_in_window = [row[0] for row in items_in_window_result.all()]

    total_items = len(urls_in_window)

    github_count = sum(1 for u in urls_in_window if classify_source(u) == "GitHub")
    hn_count = sum(1 for u in urls_in_window if classify_source(u) == "HN")
    arxiv_count = sum(1 for u in urls_in_window if classify_source(u) == "arXiv")
    rss_count = sum(1 for u in urls_in_window if classify_source(u) == "RSS")

    # ── Brain/triage: "유효" = items with trust_score IS NOT NULL
    #    (trust_score is set only when triage fully classifies an item;
    #     items that reach save_node have trust_score or use fallback ALPHA,
    #     but because the save node sets category to ALPHA as a fallback,
    #     trust_score IS NOT NULL is the best discriminator for fully-triaged
    #     items in the current schema)
    valid_result = await session.execute(
        select(func.count()).where(
            TechItem.created_at >= since,
            TechItem.trust_score.is_not(None),
        )
    )
    valid_count = valid_result.scalar_one()

    # "저장(신규)" — all items in window are "newly saved" (they wouldn't be
    # in the DB if they hadn't been saved); same as total_items for now.
    # Kept as a separate field for future differentiation.
    new_saved_count = total_items

    # ── Keep / Pass / 미분류 ───────────────────────────────────────────────
    # For tech_items created in the window, join user_assets:
    #   Keep      → AssetStatus.KEEP
    #   Pass      → AssetStatus.ARCHIVED (archiving = user decided to pass)
    #   미분류    → no user_asset row at all (+ AssetStatus.TRACKING folded in)
    #
    # DESIGN CHOICE: Tracking is folded into 미분류 because it means "still
    # watching" = not yet fully classified by the user.

    keep_result = await session.execute(
        select(func.count())
        .select_from(TechItem)
        .join(UserAsset, UserAsset.tech_id == TechItem.id, isouter=False)
        .where(
            TechItem.created_at >= since,
            UserAsset.status == AssetStatus.KEEP,
        )
    )
    keep_count = keep_result.scalar_one()

    pass_result = await session.execute(
        select(func.count())
        .select_from(TechItem)
        .join(UserAsset, UserAsset.tech_id == TechItem.id, isouter=False)
        .where(
            TechItem.created_at >= since,
            UserAsset.status == AssetStatus.ARCHIVED,
        )
    )
    pass_count = pass_result.scalar_one()

    # 미분류 = items with no user_asset row OR status=Tracking
    no_asset_result = await session.execute(
        select(func.count())
        .select_from(TechItem)
        .join(UserAsset, UserAsset.tech_id == TechItem.id, isouter=True)
        .where(
            TechItem.created_at >= since,
            (UserAsset.id.is_(None)) | (UserAsset.status == AssetStatus.TRACKING),
        )
    )
    unclassified_count = no_asset_result.scalar_one()

    # ── Portfolio: cumulative Keep count (window-independent) ────────────
    cumulative_keep_result = await session.execute(
        select(func.count()).where(UserAsset.status == AssetStatus.KEEP)
    )
    total_keep_cumulative = cumulative_keep_result.scalar_one()

    # ── Track alerts: track_history rows in the window ───────────────────
    track_alert_result = await session.execute(
        select(func.count()).where(TrackHistory.changed_at >= since)
    )
    track_alert_count = track_alert_result.scalar_one()

    return {
        "total_items": total_items,
        "github_count": github_count,
        "hn_count": hn_count,
        "rss_count": rss_count,
        "arxiv_count": arxiv_count,
        "valid_count": valid_count,
        "new_saved_count": new_saved_count,
        "keep_count": keep_count,
        "pass_count": pass_count,
        "unclassified_count": unclassified_count,
        "total_keep_cumulative": total_keep_cumulative,
        "track_alert_count": track_alert_count,
    }
