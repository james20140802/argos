"""Morning briefing over Web Push (ARG-321).

Once per scheduled ``argos brief`` run, every device that turned
notifications on gets **one** notification summarising the events that are new
since the previous push: how many, plus the title of the top one in the web
feed's recommended order (max evidence ``feed_score``, newest first on ties).
Tapping it opens ``/feed``.

* No new events → nothing is sent (and nothing is recorded).
* The "new" window starts at the last *delivered* dispatch (one with at least
  one device ok), capped at ``briefing.lookback_days`` so a device that
  subscribes after a long gap doesn't get a month in one line. Before any
  delivery it is 24h before the first attempt — or before now, with none.
* Every attempt that reached devices is recorded, failed ones too
  (``devices_ok = 0``). A failed attempt never moves the window, so if every
  device failed (Mac offline, push service down) the next run covers the same
  events again instead of silently dropping them — even when the very first
  attempt is the one that failed.
* 404/410 from a push service means the subscription is gone for good; its row
  is deleted. Other failures keep the row and are counted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.event_document import EventDocument
from argos.models.push import PushDispatch, PushSubscription
from argos.models.tech_event import TechEvent
from argos.models.tech_item import TechItem
from argos.web.push import PushTarget, send_push

logger = logging.getLogger(__name__)

FIRST_WINDOW = timedelta(hours=24)
FEED_URL = "/feed"
NOTIFICATION_TAG = "argos-morning-briefing"


@dataclass(frozen=True)
class BriefingDigest:
    event_count: int
    top_title: str

    @property
    def title(self) -> str:
        return f"오늘의 관측 {self.event_count}"

    @property
    def body(self) -> str:
        if self.event_count > 1:
            return f"{self.top_title} 외 {self.event_count - 1}건"
        return self.top_title

    def message(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "url": FEED_URL,
            "tag": NOTIFICATION_TAG,
        }


@dataclass(frozen=True)
class PushBriefingResult:
    """What one run did — printed into ``brief.log`` by the CLI."""

    window_start: datetime
    digest: Optional[BriefingDigest]
    devices: int = 0
    ok: int = 0
    failed: int = 0
    expired: int = 0

    def summary(self) -> str:
        if self.digest is None:
            return f"Push briefing: no new events since {self.window_start.isoformat()} — skipped"
        if self.devices == 0:
            return (
                f"Push briefing: {self.digest.event_count} new events "
                "but no subscribed devices — skipped"
            )
        return (
            f"Push briefing: {self.digest.event_count} events → "
            f"{self.ok}/{self.devices} devices ok, {self.failed} failed, "
            f"{self.expired} expired (removed)"
        )


async def window_start(
    session: AsyncSession, *, now: datetime, lookback_days: int
) -> datetime:
    floor = now - timedelta(days=lookback_days)
    delivered = (
        await session.execute(
            select(func.max(PushDispatch.sent_at)).where(PushDispatch.devices_ok > 0)
        )
    ).scalar_one()
    if delivered is not None:
        return max(delivered, floor)
    first_attempt = (
        await session.execute(select(func.min(PushDispatch.sent_at)))
    ).scalar_one()
    start = (first_attempt or now) - FIRST_WINDOW
    return max(start, floor)


async def build_digest(
    session: AsyncSession, *, since: datetime
) -> Optional[BriefingDigest]:
    """New live events created after ``since``; ``None`` when there are none.

    Only events with at least one evidence document count — the same rows the
    feed shows as event cards.
    """
    doc_sort = func.coalesce(TechItem.published_at, TechItem.created_at)
    stmt = (
        select(
            TechEvent.id,
            TechEvent.title.label("event_title"),
        )
        .join(EventDocument, EventDocument.event_id == TechEvent.id)
        .join(TechItem, TechItem.id == EventDocument.tech_item_id)
        .where(TechEvent.merged_into_id.is_(None), TechEvent.created_at > since)
        .group_by(TechEvent.id)
        .order_by(
            func.max(TechItem.feed_score).desc().nullslast(),
            func.max(doc_sort).desc(),
            TechEvent.id.desc(),
        )
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        return None
    top = rows[0]
    title = (top.event_title or "").strip()
    if not title:
        # Not named yet: the earliest-reported evidence title, like the feed card.
        title = (
            await session.execute(
                select(TechItem.title)
                .join(EventDocument, EventDocument.tech_item_id == TechItem.id)
                .where(EventDocument.event_id == top.id)
                .order_by(doc_sort.asc(), TechItem.id.asc())
                .limit(1)
            )
        ).scalar_one()
    return BriefingDigest(event_count=len(rows), top_title=title)


async def dispatch_push_briefing(
    session: AsyncSession,
    *,
    vapid,
    lookback_days: int,
    now: Optional[datetime] = None,
    sender=send_push,
) -> PushBriefingResult:
    """Compose today's digest and push it to every subscribed device."""
    now = now or datetime.now(timezone.utc)
    start = await window_start(session, now=now, lookback_days=lookback_days)
    digest = await build_digest(session, since=start)
    if digest is None:
        result = PushBriefingResult(window_start=start, digest=None)
        logger.info(result.summary())
        return result

    subs = (await session.execute(select(PushSubscription))).scalars().all()
    if not subs:
        result = PushBriefingResult(window_start=start, digest=digest)
        logger.info(result.summary())
        return result

    message = digest.message()
    ok_ids, expired_ids, failed = [], [], 0
    for sub in subs:
        outcome = await sender(
            PushTarget(
                endpoint=sub.endpoint, p256dh=sub.p256dh, auth=sub.auth, subject=sub.origin
            ),
            message,
            vapid,
        )
        if outcome.status == "ok":
            ok_ids.append(sub.id)
        elif outcome.status == "expired":
            expired_ids.append(sub.id)
            logger.info("push: subscription %s (%s) expired — removing", sub.id, sub.label)
        else:
            failed += 1
            logger.warning("push: send to %s (%s) failed: %s", sub.id, sub.label, outcome.detail)

    if expired_ids:
        await session.execute(
            delete(PushSubscription).where(PushSubscription.id.in_(expired_ids))
        )
    if ok_ids:
        await session.execute(
            update(PushSubscription)
            .where(PushSubscription.id.in_(ok_ids))
            .values(last_success_at=now)
        )
    session.add(
        PushDispatch(
            sent_at=now,
            event_count=digest.event_count,
            title=digest.body,
            devices_ok=len(ok_ids),
            devices_failed=failed,
            devices_expired=len(expired_ids),
        )
    )
    await session.commit()

    result = PushBriefingResult(
        window_start=start,
        digest=digest,
        devices=len(subs),
        ok=len(ok_ids),
        failed=failed,
        expired=len(expired_ids),
    )
    log = logger.info if failed == 0 else logger.warning
    log(result.summary())
    return result


async def upsert_subscription(
    session: AsyncSession,
    *,
    endpoint: str,
    p256dh: str,
    auth: str,
    label: str,
    origin: str,
) -> None:
    """Insert or refresh one device (same endpoint → same row, keys replaced)."""
    values = {"p256dh": p256dh, "auth": auth, "label": label, "origin": origin}
    stmt = pg_insert(PushSubscription).values(endpoint=endpoint, **values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[PushSubscription.endpoint],
        set_={**values, "updated_at": func.now()},
    )
    await session.execute(stmt)
    await session.commit()


async def remove_subscription(session: AsyncSession, *, endpoint: str) -> bool:
    res = await session.execute(
        delete(PushSubscription).where(PushSubscription.endpoint == endpoint)
    )
    await session.commit()
    return bool(res.rowcount)


async def has_subscription(session: AsyncSession, *, endpoint: str) -> bool:
    row = (
        await session.execute(
            select(PushSubscription.id).where(PushSubscription.endpoint == endpoint)
        )
    ).first()
    return row is not None
