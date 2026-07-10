"""Read-side service merging a Keep asset's timeline events (ARG-205).

``fetch_timeline`` merges three event sources for a single ``tech_item``'s
Keep asset into one reverse-chronological list, so the portfolio card
accordion and (later) the detail page can render "what happened to this
asset" without three separate round-trips in the template layer:

* **status** (✅) — real ``AssetStatus`` transitions logged in
  ``track_history`` (mirrors ``detail.py::_fetch_related_history``'s query
  shape, scoped to this tech_id's own user_asset only — unlike the detail
  page's similarity-expanded scope, the timeline is asset-local).
* **signal** (🔭) — ``SIGNAL_MATCHED`` alert rows, resolved to the matched
  tech_item via LEFT JOIN (mirrors
  ``detail.py::_fetch_signal_alerts``). A deleted matched item silently
  drops the row (inner-join effect — no dangling link). Legacy
  ``SUCCESSION_ALERTED`` rows written before ARG-204 threaded the successor
  id (``changed_from == 'Keep'``) can't be resolved to a specific successor,
  so they surface as a plain-text event (``title=None``).
* **succession** (🧬) — ``tech_succession`` rows where this tech_id is the
  *predecessor* (this asset's own successors), carrying ``relation_type`` +
  ``reasoning`` + the successor's link. Sort anchor is
  ``tech_succession.created_at`` (the row has no other timestamp).

All three are independent single-purpose queries (mirrors ``portfolio.py`` /
``detail.py``'s per-concern helper functions) merged and sorted in Python —
they draw from unrelated tables with no shared key to ORDER BY across in one
statement. Read-only: no schema changes, no writes.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from sqlalchemy import String, and_, cast, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from argos.models.tech_item import TechItem
from argos.models.tech_succession import RelationType, TechSuccession
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset


TimelineKind = Literal["status", "signal", "succession"]

# Deterministic tie-break for events sharing the same `changed_at` (e.g. rows
# seeded/written within the same DB transaction). Arbitrary but fixed order
# so test assertions and re-renders never flap: status, then signal, then
# succession.
_KIND_ORDER: dict[str, int] = {"status": 0, "signal": 1, "succession": 2}


@dataclass(frozen=True)
class TimelineEvent:
    """One entry in a Keep asset's merged timeline.

    Not every field is populated by every ``kind`` — e.g. ``relation_type``
    /``reasoning`` are succession-only, ``changed_from``/``changed_to`` are
    status-only. ``label`` is the render-ready human-readable line so
    ``_timeline.html`` doesn't need kind-specific string assembly beyond the
    icon.
    """

    kind: TimelineKind
    changed_at: datetime
    title: Optional[str]
    link_tech_id: Optional[uuid.UUID]
    changed_from: Optional[str]
    changed_to: Optional[str]
    relation_type: Optional[RelationType]
    reasoning: Optional[str]
    label: str


async def _fetch_status_events(
    session: AsyncSession, tech_id: uuid.UUID
) -> list[TimelineEvent]:
    """Real AssetStatus transitions for tech_id's own user_asset.

    Same ``changed_to ∈ AssetStatus values`` filter as
    ``detail.py::_fetch_related_history`` — excludes the alert-dedup
    sentinel rows (``signal_matched`` / ``succession_alerted``), which are
    handled separately by ``_fetch_signal_events``.
    """
    status_values = [s.value for s in AssetStatus]
    stmt = (
        select(
            TrackHistory.changed_from,
            TrackHistory.changed_to,
            TrackHistory.changed_at,
        )
        .join(UserAsset, UserAsset.id == TrackHistory.user_asset_id)
        .where(UserAsset.tech_id == tech_id)
        .where(TrackHistory.changed_to.in_(status_values))
    )
    rows = (await session.execute(stmt)).all()
    return [
        TimelineEvent(
            kind="status",
            changed_at=row.changed_at,
            title=None,
            link_tech_id=None,
            changed_from=row.changed_from,
            changed_to=row.changed_to,
            relation_type=None,
            reasoning=None,
            label=f"{row.changed_from} → {row.changed_to}",
        )
        for row in rows
    ]


async def _fetch_signal_events(
    session: AsyncSession, tech_id: uuid.UUID
) -> list[TimelineEvent]:
    """SIGNAL_MATCHED + SUCCESSION_ALERTED alert rows for tech_id's asset.

    Mirrors ``detail.py::_fetch_signal_alerts``'s LEFT JOIN shape: a
    ``signal_matched`` row's ``changed_from`` is the matched item's UUID,
    resolved back to its title/id. A miss (deleted item) drops the row
    entirely — an inner-join effect achieved by filtering out unmatched rows
    in Python, since SQLAlchemy has already produced the LEFT JOIN's NULLs
    for us to filter on.
    """
    from argos.slack.services.track_check import SIGNAL_MATCHED, SUCCESSION_ALERTED

    Matched = aliased(TechItem)
    stmt = (
        select(
            TrackHistory.changed_to,
            TrackHistory.changed_from,
            TrackHistory.changed_at,
            Matched.id.label("matched_id"),
            Matched.title.label("matched_title"),
        )
        .join(UserAsset, UserAsset.id == TrackHistory.user_asset_id)
        .join(
            Matched,
            and_(
                TrackHistory.changed_to == SIGNAL_MATCHED,
                cast(Matched.id, String) == TrackHistory.changed_from,
            ),
            isouter=True,
        )
        .where(UserAsset.tech_id == tech_id)
        .where(TrackHistory.changed_to.in_((SIGNAL_MATCHED, SUCCESSION_ALERTED)))
    )
    rows = (await session.execute(stmt)).all()

    events: list[TimelineEvent] = []
    for row in rows:
        if row.changed_to == SIGNAL_MATCHED:
            if row.matched_id is None:
                # Matched item was deleted since the alert fired — the link
                # target is gone, so the row is silently excluded rather than
                # rendered with a dead/empty link.
                continue
            events.append(
                TimelineEvent(
                    kind="signal",
                    changed_at=row.changed_at,
                    title=row.matched_title,
                    link_tech_id=row.matched_id,
                    changed_from=row.changed_from,
                    changed_to=row.changed_to,
                    relation_type=None,
                    reasoning=None,
                    label=f"새 신호: {row.matched_title}",
                )
            )
        else:  # SUCCESSION_ALERTED
            # Legacy rows (pre-ARG-204) recorded `changed_from='Keep'`
            # rather than the successor id, so the specific successor can't
            # be resolved here — surface as a plain-text event.
            events.append(
                TimelineEvent(
                    kind="signal",
                    changed_at=row.changed_at,
                    title=None,
                    link_tech_id=None,
                    changed_from=row.changed_from,
                    changed_to=row.changed_to,
                    relation_type=None,
                    reasoning=None,
                    label="후속 기술 신호",
                )
            )
    return events


async def _fetch_succession_events(
    session: AsyncSession, tech_id: uuid.UUID
) -> list[TimelineEvent]:
    """tech_succession rows where tech_id is the predecessor (its successors)."""
    Succ = aliased(TechItem)
    stmt = (
        select(
            Succ.id.label("succ_id"),
            Succ.title.label("succ_title"),
            TechSuccession.relation_type,
            TechSuccession.reasoning,
            TechSuccession.created_at,
        )
        .join(Succ, Succ.id == TechSuccession.successor_id)
        .where(TechSuccession.predecessor_id == tech_id)
    )
    rows = (await session.execute(stmt)).all()
    return [
        TimelineEvent(
            kind="succession",
            changed_at=row.created_at,
            title=row.succ_title,
            link_tech_id=row.succ_id,
            changed_from=None,
            changed_to=None,
            relation_type=row.relation_type,
            reasoning=row.reasoning,
            label=f"{row.relation_type.value}: {row.succ_title}",
        )
        for row in rows
    ]


async def fetch_timeline(
    session: AsyncSession,
    tech_id: uuid.UUID,
    *,
    limit: Optional[int] = None,
) -> list[TimelineEvent]:
    """Merge status/signal/succession events for ``tech_id``, newest first.

    ``limit=None`` returns the full history (detail page); ``limit=5``
    returns only the 5 most recent (portfolio card accordion). Ties on
    ``changed_at`` break deterministically via ``_KIND_ORDER``.
    """
    status_events = await _fetch_status_events(session, tech_id)
    signal_events = await _fetch_signal_events(session, tech_id)
    succession_events = await _fetch_succession_events(session, tech_id)

    events = status_events + signal_events + succession_events

    # Two-pass stable sort: first the deterministic tie-break (ascending),
    # then changed_at descending. Python's sort is stable, so equal-timestamp
    # events keep their _KIND_ORDER relative order after the second pass.
    events.sort(key=lambda e: _KIND_ORDER[e.kind])
    events.sort(key=lambda e: e.changed_at, reverse=True)

    if limit is not None:
        events = events[:limit]
    return events
