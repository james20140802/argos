"""Succession alert checks (ARG-56, ARG-204) and signal-match checks (ARG-115).

**Succession alerts** find succession records whose predecessor is currently
held as a Keep-ed user_asset, skipping any (user_asset, successor) pair that
has already received a succession alert (recorded in ``track_history`` with
``changed_to = 'succession_alerted'`` and ``changed_from = str(successor_id)``
— see ARG-204).  Dedup is per pair, not per asset: a Keep-ed asset that
already alerted for one successor still alerts for a *different* successor.

**Signal match** (ARG-115) compares the embeddings of newly-saved TechItems
against Keep-ed assets using pgvector cosine similarity, reporting matches
above the 0.85 threshold that have not yet been notified (i.e. no
``track_history`` row with ``changed_to = SIGNAL_MATCHED`` and
``changed_from = str(new_item_id)`` for the (user_asset_id, new_item_id)
pair).

By default, all ``tech_succession`` rows are scanned — not just those whose
successor was saved in the current pipeline run.  This is intentional: if
``post_track_update`` fails to deliver an alert (transient Slack outage, rate
limit, etc.), no ``track_history`` row is written, and on the next run the
same successor will already exist in ``tech_items`` (deduplicated by
``source_url`` in ``save_node``) so it would never re-enter a "new this run"
candidate set.  Scanning all unalerted rows guarantees the alert is retried.

Public surface:
- :class:`SuccessionAlert` — value object handed to the Slack layer.
- :func:`check_succession` — pure async DB query, no side effects.
- :class:`SignalMatch` — value object for signal-match results (ARG-115).
- :func:`match_signals` — pgvector cosine similarity query (ARG-115).
- :func:`post_signal_update` — Slack dispatcher + track_history logger (ARG-116).

The Slack dispatcher (ARG-104) is responsible for posting messages and writing
the ``track_history`` row that marks an alert as delivered.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import timezone

from sqlalchemy import String, and_, cast, select, text
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

# Sentinel written to ``track_history.changed_to`` when a signal-match alert
# is delivered.  Kept module-local to keep the query predicate and the writer
# in sync.  ``changed_from`` stores ``str(new_item_id)`` (36 chars, UUID hex)
# so the dedup predicate can target the exact (user_asset_id, new_item_id) pair
# without exceeding the String(50) column limit on either field.
SIGNAL_MATCHED = "signal_matched"

# Cosine similarity threshold above which a new TechItem is considered a
# "follow-up signal" for a Keep-ed asset.  1 − cosine_distance > threshold.
# ARG-204: this is now only the *default* value (mirrored by
# TrackingConfig.signal_similarity_threshold in argos.config). At runtime,
# match_signals() reads the effective threshold from
# settings.user.tracking.signal_similarity_threshold instead of this
# constant directly.
SIGNAL_SIMILARITY_THRESHOLD = 0.85


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
    successor_id:
        tech_items.id of the successor.  ARG-204: used to dedup alerts per
        (user_asset, successor) pair — mirrors ``SignalMatch.new_item_id``.
        Added last so existing keyword-based construction call sites are
        unaffected.
    """

    user_asset_id: uuid.UUID
    predecessor_title: str
    successor_title: str
    relation_type: RelationType
    successor_id: uuid.UUID


async def check_succession(
    session: AsyncSession,
    new_item_ids: list[uuid.UUID] | None = None,
) -> list[SuccessionAlert]:
    """Return succession alerts for Keep-ed predecessors with un-alerted successors.

    Parameters
    ----------
    session:
        Async SQLAlchemy session bound to the Argos DB.
    new_item_ids:
        Optional list of tech_item UUIDs to narrow the candidate
        ``successor_id`` set to.  When ``None`` (default), **all**
        ``tech_succession`` rows are considered — this is the correct mode
        for the periodic pipeline because it lets failed Slack sends from
        previous runs be retried.  When an explicit list is supplied, only
        successors in that list are considered (useful for tests or
        targeted re-checks).  An empty list short-circuits to ``[]``.

    Returns
    -------
    list[SuccessionAlert]
        One alert per (Keep-ed predecessor, successor) pair that has not yet
        had a ``succession_alerted`` row written to ``track_history`` for the
        matching ``(user_asset, successor)`` pair.

    Notes
    -----
    Dedup granularity (ARG-204): alerts are deduplicated per
    ``(user_asset_id, successor_id)`` pair — mirroring :func:`match_signals`'
    ``(user_asset_id, new_item_id)`` encoding.  Once a succession alert has
    been recorded for a specific (Keep-ed asset, successor) pair, only that
    exact pair is suppressed; a *different* successor for the same Keep-ed
    asset still alerts.  The pair is encoded the same way
    :func:`post_signal_update` encodes signal matches:
    ``changed_from = str(successor_id)``, ``changed_to = SUCCESSION_ALERTED``
    — no schema change required.

    Retry semantics: alerts that were generated but failed to post (no
    ``track_history`` row written by :func:`post_track_update`) will reappear
    in subsequent calls until a successful post records them.  See module
    docstring for the full rationale.
    """
    # Explicit empty list = caller asked us to look at nothing.
    if new_item_ids is not None and len(new_item_ids) == 0:
        return []

    Predecessor = aliased(TechItem)
    Successor = aliased(TechItem)

    # NOT EXISTS subquery: skip (user_asset, successor) pairs that already
    # have a `succession_alerted` row in track_history for that exact pair.
    # ARG-204: correlated on successor_id (cast to text) in addition to
    # user_asset_id — mirrors match_signals' NOT EXISTS predicate
    # (`th.changed_from = CAST(ni.id AS text)`) and detail.py's
    # `cast(Matched.id, String) == TrackHistory.changed_from` join condition.
    already_alerted = (
        select(TrackHistory.id)
        .where(
            and_(
                TrackHistory.user_asset_id == UserAsset.id,
                TrackHistory.changed_to == SUCCESSION_ALERTED,
                TrackHistory.changed_from == cast(TechSuccession.successor_id, String),
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
            TechSuccession.successor_id.label("successor_id"),
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
            UserAsset.status == AssetStatus.KEEP,
            ~already_alerted,
        )
        .order_by(TechSuccession.created_at.asc())
    )

    # Optional narrowing for callers (e.g. tests) that want to restrict the
    # scan to a known set of successor IDs.  Production pipeline passes None.
    if new_item_ids is not None:
        stmt = stmt.where(TechSuccession.successor_id.in_(new_item_ids))

    result = await session.execute(stmt)

    # In-batch dedup: the track_history NOT EXISTS predicate above only
    # filters pairs that were alerted in *prior committed* runs.  Within a
    # single query result, duplicate/overlapping tech_succession rows for
    # the exact same (asset, successor) pair would otherwise yield more than
    # one alert for that pair — and because ``post_track_update`` writes its
    # dedup ``track_history`` row only after each Slack send (within the
    # same un-committed session), the NOT EXISTS predicate can't see those
    # in-flight markers either.  ARG-204: collapse to one alert per
    # (asset_id, successor_id) pair here — NOT per asset_id alone, so a
    # Keep-ed asset with multiple *distinct* successors still yields one
    # alert per successor.  Representative rule: **first encountered**,
    # which — given the ``ORDER BY TechSuccession.created_at ASC`` above —
    # is the earliest-created succession row.  This is deterministic and
    # matches the per-pair dedup contract documented in the Notes section
    # above.
    seen_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    alerts: list[SuccessionAlert] = []
    for row in result.all():
        asset_id = row[0]
        successor_id = row[4]
        pair = (asset_id, successor_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        alerts.append(
            SuccessionAlert(
                user_asset_id=asset_id,
                predecessor_title=row[1],
                successor_title=row[2],
                relation_type=row[3],
                successor_id=successor_id,
            )
        )
    return alerts


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

        # Mark this (user_asset, successor) pair as delivered so future runs
        # skip only that exact pair.  ARG-204: ``changed_from`` now encodes
        # ``str(alert.successor_id)`` (36 chars, fits String(50)) instead of
        # the legacy asset-level ``'Keep'`` literal — mirrors
        # post_signal_update's ``changed_from=str(match.new_item_id)``
        # encoding.  ``changed_to`` remains the sentinel string that
        # check_succession's NOT EXISTS predicate looks for.  Pre-ARG-204
        # rows still carrying ``changed_from='Keep'`` match no successor
        # UUID under the new predicate, so those assets naturally re-alert
        # once — intended, not a bug (see check_succession's Notes).
        session.add(
            TrackHistory(
                user_asset_id=alert.user_asset_id,
                changed_from=str(alert.successor_id),
                changed_to=SUCCESSION_ALERTED,
            )
        )


# ---------------------------------------------------------------------------
# Signal-match — ARG-115 / ARG-116
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalMatch:
    """A single signal-match result ready for Slack dispatch.

    Attributes
    ----------
    user_asset_id:
        ID of the Keep-ed user_asset whose tech item matched.
    keep_item_id:
        tech_item.id of the Keep-ed asset.
    keep_item_title:
        Title of the Keep-ed tech item.
    new_item_id:
        tech_item.id of the newly-saved item that matched.
    new_item_title:
        Title of the new tech item.
    new_item_url:
        Source URL of the new tech item (for Slack linking).
    similarity_score:
        1 − cosine_distance; value in (0, 1].
    """

    user_asset_id: uuid.UUID
    keep_item_id: uuid.UUID
    keep_item_title: str
    new_item_id: uuid.UUID
    new_item_title: str
    new_item_url: str
    similarity_score: float


async def match_signals(
    session: AsyncSession,
    new_item_ids: list[uuid.UUID] | None = None,
) -> list[SignalMatch]:
    """Return signal matches: new TechItems similar to Keep-ed assets.

    Uses pgvector cosine distance (``<=>``) to compare embeddings.
    Similarity is defined as ``1 − cosine_distance``; only pairs above
    :data:`SIGNAL_SIMILARITY_THRESHOLD` (0.85) are returned.

    Dedup granularity is per ``(user_asset_id, new_item_id)`` pair — the
    same Keep-ed asset can re-alert for different new items.  A pair is
    suppressed only when ``track_history`` already contains a row with
    ``user_asset_id = <asset_id>``, ``changed_to = SIGNAL_MATCHED``, and
    ``changed_from = str(new_item_id)`` (the UUID written by
    :func:`post_signal_update` after a successful Slack send).

    Parameters
    ----------
    session:
        Async SQLAlchemy session bound to the Argos DB.
    new_item_ids:
        Optional list of tech_item UUIDs to consider as candidates.
        When ``None``, *all* tech_items are candidates (full retry scan).
        When an explicit list is given, only those items are compared.
        An empty list short-circuits to ``[]``.

    Returns
    -------
    list[SignalMatch]
        One entry per (Keep-ed asset, new item) pair that passes the
        threshold and has not yet been notified.  No ordering guarantee.
    """
    if new_item_ids is not None and len(new_item_ids) == 0:
        return []

    # Build the NOT-ALREADY-NOTIFIED filter as a SQL fragment.
    # dedup key: (user_asset_id, changed_to='signal_matched', changed_from=str(new_item_id))
    # We use a NOT EXISTS subquery correlated on both user_asset_id and new_item_id.
    #
    # Using raw SQL (same as search.py) for the pgvector <=> operator because
    # SQLAlchemy's type system does not natively expose the pgvector operators.

    base_sql = """
        SELECT
            ua.id          AS user_asset_id,
            ki.id          AS keep_item_id,
            ki.title       AS keep_item_title,
            ni.id          AS new_item_id,
            ni.title       AS new_item_title,
            ni.source_url  AS new_item_url,
            1.0 - (ki.embedding <=> ni.embedding) AS similarity_score
        FROM user_assets ua
        JOIN tech_items ki ON ki.id = ua.tech_id
        CROSS JOIN tech_items ni
        WHERE
            ua.status = 'Keep'
            AND ki.embedding IS NOT NULL
            AND ni.embedding IS NOT NULL
            AND ki.id != ni.id
            AND (1.0 - (ki.embedding <=> ni.embedding)) > :threshold
            AND NOT EXISTS (
                SELECT 1 FROM track_history th
                WHERE th.user_asset_id = ua.id
                  AND th.changed_to    = :sentinel
                  AND th.changed_from  = CAST(ni.id AS text)
            )
    """

    # ARG-204: the threshold is config-driven (settings.user.tracking.
    # signal_similarity_threshold). Lazy-imported so this module (already
    # imported at CLI/pipeline startup) doesn't force-load the full config
    # singleton at import time — mirrors the lazy `from argos.slack.blocks
    # import ...` pattern used elsewhere in this file.
    from argos.config import settings

    threshold = settings.user.tracking.signal_similarity_threshold

    params: dict = {
        "threshold": threshold,
        "sentinel": SIGNAL_MATCHED,
    }

    if new_item_ids is not None:
        # Build a parameterised IN-list.
        placeholders = ", ".join(
            f":nid_{i}" for i in range(len(new_item_ids))
        )
        base_sql += f"  AND ni.id IN ({placeholders})\n"
        for i, nid in enumerate(new_item_ids):
            params[f"nid_{i}"] = str(nid)

    result = await session.execute(text(base_sql), params)
    rows = result.fetchall()

    return [
        SignalMatch(
            user_asset_id=uuid.UUID(str(row.user_asset_id)),
            keep_item_id=uuid.UUID(str(row.keep_item_id)),
            keep_item_title=row.keep_item_title,
            new_item_id=uuid.UUID(str(row.new_item_id)),
            new_item_title=row.new_item_title,
            new_item_url=row.new_item_url,
            similarity_score=float(row.similarity_score),
        )
        for row in rows
    ]


async def post_signal_update(
    app,
    channel: str,
    matches: list[SignalMatch],
    session: AsyncSession,
) -> None:
    """Post each signal match to Slack and record a track_history sentinel row.

    One ``chat_postMessage`` is issued per match.  After a successful post,
    a ``TrackHistory`` row is added with:
    - ``user_asset_id``: the Keep-ed asset
    - ``changed_to   ``: ``SIGNAL_MATCHED`` (14 chars)
    - ``changed_from ``: ``str(new_item_id)`` (36 chars UUID)

    This encoding keeps both columns within the ``String(50)`` limit and
    lets the NOT EXISTS predicate in :func:`match_signals` target the exact
    ``(user_asset_id, new_item_id)`` pair.

    If Slack fails for a single match, the failure is logged, no history
    row is written (so the match is retried next run), and processing
    continues with the remaining matches.  ``user_assets.last_monitored_at``
    is updated only for successfully-notified assets.

    The session is **not** committed here; the caller (CLI ``_run``) owns
    the commit lifecycle.

    Parameters
    ----------
    app:
        ``slack_bolt.async_app.AsyncApp`` (or any object with an
        ``.client.chat_postMessage`` async method).  Duck-typed for tests.
    channel:
        Slack channel ID.
    matches:
        Results from :func:`match_signals`.  Empty list is a no-op.
    session:
        Active async SQLAlchemy session.
    """
    from datetime import datetime

    from argos.slack.blocks import build_signal_match_blocks

    if not matches:
        return

    for match in matches:
        blocks = build_signal_match_blocks(match)
        fallback = (
            f"🔭 Keep한 {match.keep_item_title}과 유사한 신호: "
            f"{match.new_item_title} (유사도 {match.similarity_score:.0%})"
        )
        try:
            await app.client.chat_postMessage(
                channel=channel,
                blocks=blocks,
                text=fallback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "post_signal_update: chat_postMessage failed for asset %s / new_item %s: %r",
                match.user_asset_id, match.new_item_id, exc,
            )
            continue

        # Mark this (user_asset_id, new_item_id) pair as notified.
        # changed_from = str(new_item_id) (36 chars)
        # changed_to   = SIGNAL_MATCHED   (14 chars)
        session.add(
            TrackHistory(
                user_asset_id=match.user_asset_id,
                changed_from=str(match.new_item_id),
                changed_to=SIGNAL_MATCHED,
            )
        )

        # Update last_monitored_at on the UserAsset.
        from sqlalchemy import update as sa_update

        now = datetime.now(tz=timezone.utc)
        await session.execute(
            sa_update(UserAsset)
            .where(UserAsset.id == match.user_asset_id)
            .values(last_monitored_at=now),
        )
