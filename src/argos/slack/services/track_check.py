"""Succession alert checks (ARG-56).

Given a list of newly-saved tech_item IDs, find succession records whose
predecessor is currently held as a Keep-ed user_asset, skipping any Keep-ed
asset that has already received a succession alert (recorded in ``track_history``
with ``changed_to = 'succession_alerted'``).

Public surface:
- :class:`SuccessionAlert` — value object handed to the Slack layer.
- :func:`check_succession` — pure async DB query, no side effects.

The Slack dispatcher (ARG-104) is responsible for posting messages and writing
the ``track_history`` row that marks an alert as delivered.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from argos.models.tech_item import TechItem
from argos.models.tech_succession import RelationType, TechSuccession
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset

logger = logging.getLogger(__name__)

# Sentinel string written to ``track_history.changed_to`` to mark that a
# succession alert has been delivered for a Keep-ed asset.  Kept module-local
# so the Slack dispatcher (ARG-104) and the dedup predicate stay in sync.
SUCCESSION_ALERTED = "succession_alerted"


@dataclass(frozen=True)
class SuccessionAlert:
    """A single succession alert ready for Slack dispatch.

    Attributes
    ----------
    user_asset_id:
        ID of the Keep-ed user_asset whose tech is being superseded.  The
        dispatcher writes a track_history row keyed on this ID after posting.
    predecessor_title:
        Title of the older tech (the one the user is Keep-ing).
    successor_title:
        Title of the newly-saved tech that supersedes the predecessor.
    relation_type:
        Whether the successor Replaces, Enhances, or Forks the predecessor.
    """

    user_asset_id: uuid.UUID
    predecessor_title: str
    successor_title: str
    relation_type: RelationType


async def check_succession(
    session: AsyncSession,
    new_item_ids: list[uuid.UUID],
) -> list[SuccessionAlert]:
    """Return succession alerts for newly-saved items whose predecessor is Keep-ed.

    Parameters
    ----------
    session:
        Async SQLAlchemy session bound to the Argos DB.
    new_item_ids:
        UUIDs of tech_items that were just saved by ``save_node``.  These are
        candidate ``successor_id`` values to look up in ``tech_succession``.

    Returns
    -------
    list[SuccessionAlert]
        One alert per (Keep-ed predecessor, successor) pair that has not yet
        had a ``succession_alerted`` row written to ``track_history`` for the
        matching ``user_asset``.

    Notes
    -----
    Empty input short-circuits to ``[]`` without issuing a query.

    Dedup granularity: alerts are deduplicated per ``user_asset_id``.  Once any
    succession alert has been recorded for a Keep-ed asset, subsequent alerts
    for the *same* Keep-ed asset are suppressed — even if the new alert is
    about a different successor.  This matches the spec in ARG-104 and avoids
    schema changes to ``track_history``.
    """
    if not new_item_ids:
        return []

    Predecessor = aliased(TechItem)
    Successor = aliased(TechItem)

    # NOT EXISTS subquery: skip user_assets that already have any
    # `succession_alerted` row in track_history.
    already_alerted = (
        select(TrackHistory.id)
        .where(
            and_(
                TrackHistory.user_asset_id == UserAsset.id,
                TrackHistory.changed_to == SUCCESSION_ALERTED,
            )
        )
        .exists()
    )

    stmt = (
        select(
            UserAsset.id.label("user_asset_id"),
            Predecessor.title.label("predecessor_title"),
            Successor.title.label("successor_title"),
            TechSuccession.relation_type.label("relation_type"),
        )
        .select_from(TechSuccession)
        .join(
            UserAsset,
            UserAsset.tech_id == TechSuccession.predecessor_id,
        )
        .join(
            Predecessor,
            Predecessor.id == TechSuccession.predecessor_id,
        )
        .join(
            Successor,
            Successor.id == TechSuccession.successor_id,
        )
        .where(
            TechSuccession.successor_id.in_(new_item_ids),
            UserAsset.status == AssetStatus.KEEP,
            ~already_alerted,
        )
        .order_by(TechSuccession.created_at.asc())
    )

    result = await session.execute(stmt)
    return [
        SuccessionAlert(
            user_asset_id=row[0],
            predecessor_title=row[1],
            successor_title=row[2],
            relation_type=row[3],
        )
        for row in result.all()
    ]


async def post_track_update(
    app,
    channel: str,
    alerts: list[SuccessionAlert],
    session: AsyncSession,
) -> None:
    """Post each succession alert to Slack and record a track_history row.

    Per ARG-104, one ``chat_postMessage`` call is issued per alert.  After a
    successful post, a ``TrackHistory`` row is added with
    ``changed_to = 'succession_alerted'`` so that ``check_succession`` skips
    the same Keep-ed asset on subsequent runs.  The session is not committed
    here — the caller (CLI ``_run``) owns the commit lifecycle.

    If Slack fails on a single alert (network error, rate limit, etc.), the
    failure is logged and the remaining alerts are still attempted.  No
    history row is written for a failed send, so the alert remains eligible
    on the next run.

    Parameters
    ----------
    app:
        ``slack_bolt.async_app.AsyncApp`` (or any object with an
        ``.client.chat_postMessage`` async method).  Duck-typed to keep the
        unit tests free of a real Slack dependency.
    channel:
        Slack channel ID (typically ``settings.user.slack.channel_id``).
    alerts:
        Alerts produced by :func:`check_succession`.  Empty list is a no-op.
    session:
        Active async SQLAlchemy session.  ``TrackHistory`` rows are added but
        not committed; the caller commits.
    """
    # Imported lazily to avoid a circular import (blocks.py is small but is
    # imported by briefing.py which transitively pulls in track_check via
    # potential future re-export paths).
    from argos.slack.blocks import build_succession_alert_blocks

    if not alerts:
        return

    for alert in alerts:
        blocks = build_succession_alert_blocks(alert)
        fallback = (
            f"⚠️ Keep한 {alert.predecessor_title}을 대체하는 "
            f"{alert.successor_title}이 등장했습니다"
        )
        try:
            await app.client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=fallback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "post_track_update: chat_postMessage failed for asset %s: %r",
                alert.user_asset_id, exc,
            )
            continue

        # Mark the alert as delivered so future runs skip it.
        # ``changed_from='Keep'`` — only Keep-ed assets trigger succession
        # alerts, so the previous status is always Keep.  ``changed_to`` is
        # the sentinel string that the check_succession NOT EXISTS predicate
        # looks for.
        session.add(
            TrackHistory(
                user_asset_id=alert.user_asset_id,
                changed_from=AssetStatus.KEEP.value,
                changed_to=SUCCESSION_ALERTED,
            )
        )
