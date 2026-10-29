"""ARG-321: morning push digest — DB-backed selection and dispatch bookkeeping.

Self-skips without Postgres (release CI). Seeded events carry ``created_at``
in 2099 and the digest window starts in 2098, so whatever else lives in the
scratch DB never leaks into the count. Push tables are emptied around every
test (they only ever hold this file's rows in the scratch DB).
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from argos.config import settings
from argos.web.push import SendOutcome
from argos.web.services.push_briefing import (
    FIRST_WINDOW,
    build_digest,
    dispatch_push_briefing,
    has_subscription,
    remove_subscription,
    upsert_subscription,
    window_start,
)
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url

pytestmark = [
    pytest.mark.skipif(
        not _db_reachable(_DB_URL),
        reason="pgvector DB not reachable — skipping ARG-321 DB-backed tests",
    ),
    pytest.mark.asyncio,
]

BASE = datetime(2099, 1, 1, tzinfo=timezone.utc)
SINCE = BASE - timedelta(days=365)


@asynccontextmanager
async def _db():
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.event_document import EventDocument
    from argos.models.push import PushDispatch, PushSubscription
    from argos.models.tech_event import TechEvent
    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    items: list[uuid.UUID] = []
    events: list[uuid.UUID] = []

    async def wipe_push():
        async with Session() as s:
            await s.execute(delete(PushDispatch))
            await s.execute(delete(PushSubscription))
            await s.commit()

    class Seed:
        async def event(self, s, *, title, hours=0.0, scores=(None,), merged_into=None):
            ev = TechEvent(
                title=title,
                occurred_at=BASE + timedelta(hours=hours),
                created_at=BASE + timedelta(hours=hours),
                merged_into_id=merged_into,
            )
            s.add(ev)
            await s.flush()
            events.append(ev.id)
            for i, score in enumerate(scores):
                doc = TechItem(
                    title=f"doc {title} {i}",
                    source_url=f"https://arg321.example/{uuid.uuid4()}",
                    raw_content="x",
                    category=CategoryType.MAINSTREAM,
                    published_at=BASE + timedelta(hours=hours, minutes=i),
                    feed_score=score,
                )
                s.add(doc)
                await s.flush()
                items.append(doc.id)
                s.add(EventDocument(event_id=ev.id, tech_item_id=doc.id))
            await s.flush()
            return ev

    await wipe_push()
    try:
        yield Session, Seed()
    finally:
        async with Session() as s:
            if events:
                await s.execute(delete(EventDocument).where(EventDocument.event_id.in_(events)))
                # Tombstones point at survivors (RESTRICT): clear pointers first.
                from sqlalchemy import update

                await s.execute(
                    update(TechEvent).where(TechEvent.id.in_(events)).values(merged_into_id=None)
                )
                await s.execute(delete(TechEvent).where(TechEvent.id.in_(events)))
            if items:
                await s.execute(delete(TechItem).where(TechItem.id.in_(items)))
            await s.commit()
        await wipe_push()
        await engine.dispose()


def _sender(outcomes):
    """Fake ``send_push``: answers from ``outcomes`` by endpoint, records calls."""
    calls = []

    async def send(target, message, vapid):
        calls.append((target.endpoint, message))
        return SendOutcome(outcomes[target.endpoint])

    send.calls = calls
    return send


async def test_digest_counts_new_live_events_and_picks_top_score():
    async with _db() as (Session, seed):
        async with Session() as s:
            survivor = await seed.event(s, title="Top event", hours=1, scores=(0.2, 0.9))
            await seed.event(s, title="Other", hours=2, scores=(0.5,))
            await seed.event(s, title="Unscored newest", hours=3, scores=(None,))
            await seed.event(s, title="Merged away", hours=4, scores=(0.99,), merged_into=survivor.id)
            await s.commit()

            digest = await build_digest(s, since=SINCE)
            assert digest is not None
            assert digest.event_count == 3  # the tombstone is not an event anymore
            assert digest.top_title == "Top event"
            assert digest.body == "Top event 외 2건"

            assert await build_digest(s, since=BASE + timedelta(hours=10)) is None


async def test_digest_title_falls_back_to_earliest_document():
    async with _db() as (Session, seed):
        async with Session() as s:
            await seed.event(s, title=None, hours=1, scores=(0.5, 0.4))
            await s.commit()
            digest = await build_digest(s, since=SINCE)
            assert digest.top_title == "doc None 0"


async def test_dispatch_sends_once_per_device_and_cleans_up_expired():
    async with _db() as (Session, seed):
        async with Session() as s:
            await seed.event(s, title="A", hours=1, scores=(0.5,))
            await seed.event(s, title="B", hours=2, scores=(0.1,))
            await s.commit()
            for name in ("ok", "gone", "flaky"):
                await upsert_subscription(
                    s, endpoint=f"https://push.example/{name}", p256dh="k", auth="a", label=name,
                    origin="https://mac.example.ts.net",
                )

            send = _sender({
                "https://push.example/ok": "ok",
                "https://push.example/gone": "expired",
                "https://push.example/flaky": "failed",
            })
            now = BASE + timedelta(hours=5)
            # Pretend the last dispatch was long ago so the 2099 events are in-window.
            result = await dispatch_push_briefing(
                s, vapid=None, lookback_days=100_000, now=now, sender=send
            )

            assert (result.devices, result.ok, result.failed, result.expired) == (3, 1, 1, 1)
            assert len(send.calls) == 3
            assert {m["body"] for _, m in send.calls} == {"A 외 1건"}
            assert all(m["url"] == "/feed" for _, m in send.calls)
            assert not await has_subscription(s, endpoint="https://push.example/gone")
            assert await has_subscription(s, endpoint="https://push.example/flaky")

            # The recorded dispatch moves the window: nothing new → nothing sent.
            assert await window_start(s, now=now, lookback_days=100_000) == now
            again = await dispatch_push_briefing(
                s, vapid=None, lookback_days=100_000, now=now + timedelta(days=1), sender=send
            )
            assert again.digest is None
            assert len(send.calls) == 3


async def test_failed_first_dispatch_keeps_its_events_for_the_retry():
    """Monday's very first push fails on every device; Tuesday's run must still
    carry Monday's event (the window must not reset to "last 24h")."""
    async with _db() as (Session, seed):
        async with Session() as s:
            await seed.event(s, title="Monday news", hours=1, scores=(0.5,))
            await s.commit()
            await upsert_subscription(
                s,
                endpoint="https://push.example/dev",
                p256dh="k",
                auth="a",
                label="Mac",
                origin="https://mac.example.ts.net",
            )
            monday = BASE + timedelta(hours=5)
            tuesday = monday + timedelta(days=1)

            down = _sender({"https://push.example/dev": "failed"})
            first = await dispatch_push_briefing(
                s, vapid=None, lookback_days=7, now=monday, sender=down
            )
            assert first.ok == 0 and first.failed == 1

            # Window still opens 24h before the failed first attempt.
            assert await window_start(s, now=tuesday, lookback_days=7) == monday - FIRST_WINDOW

            up = _sender({"https://push.example/dev": "ok"})
            retry = await dispatch_push_briefing(
                s, vapid=None, lookback_days=7, now=tuesday, sender=up
            )
            assert retry.ok == 1
            assert retry.digest is not None and retry.digest.top_title == "Monday news"
            # Delivered: the window now starts at Tuesday's dispatch.
            assert await window_start(s, now=tuesday, lookback_days=7) == tuesday


async def test_failed_attempt_never_moves_window_after_a_delivery():
    async with _db() as (Session, seed):
        async with Session() as s:
            await seed.event(s, title="A", hours=1, scores=(0.5,))
            await s.commit()
            await upsert_subscription(
                s,
                endpoint="https://push.example/dev",
                p256dh="k",
                auth="a",
                label="Mac",
                origin="https://mac.example.ts.net",
            )
            t0 = BASE - timedelta(hours=10)
            await dispatch_push_briefing(
                s, vapid=None, lookback_days=7, now=t0,
                sender=_sender({"https://push.example/dev": "ok"}),
            )
            t1 = BASE + timedelta(hours=5)
            await dispatch_push_briefing(
                s, vapid=None, lookback_days=7, now=t1,
                sender=_sender({"https://push.example/dev": "failed"}),
            )
            assert await window_start(s, now=t1 + timedelta(days=1), lookback_days=7) == t0


async def test_window_is_capped_by_lookback():
    async with _db() as (Session, seed):
        async with Session() as s:
            await seed.event(s, title="A", hours=1, scores=(0.5,))
            await s.commit()
            await upsert_subscription(
                s,
                endpoint="https://push.example/dev",
                p256dh="k",
                auth="a",
                label="Mac",
                origin="https://mac.example.ts.net",
            )
            t0 = BASE + timedelta(hours=5)
            await dispatch_push_briefing(
                s, vapid=None, lookback_days=7, now=t0,
                sender=_sender({"https://push.example/dev": "failed"}),
            )
            later = t0 + timedelta(days=30)
            assert await window_start(s, now=later, lookback_days=7) == later - timedelta(days=7)


async def test_window_defaults_to_24h_without_a_previous_dispatch():
    async with _db() as (Session, _seed):
        async with Session() as s:
            now = BASE
            assert await window_start(s, now=now, lookback_days=7) == now - FIRST_WINDOW


async def test_no_devices_sends_nothing():
    async with _db() as (Session, seed):
        async with Session() as s:
            await seed.event(s, title="A", hours=1, scores=(0.5,))
            await s.commit()
            send = _sender({})
            result = await dispatch_push_briefing(
                s, vapid=None, lookback_days=100_000, now=BASE + timedelta(hours=5), sender=send
            )
            assert result.digest is not None and result.devices == 0
            assert send.calls == []


async def test_subscription_upsert_and_remove():
    async with _db() as (Session, _seed):
        async with Session() as s:
            ep = "https://push.example/dev"
            origin = "https://mac.example.ts.net"
            await upsert_subscription(
                s, endpoint=ep, p256dh="k1", auth="a1", label="iPhone", origin=origin
            )
            await upsert_subscription(
                s, endpoint=ep, p256dh="k2", auth="a2", label="iPhone", origin=origin
            )
            from sqlalchemy import select

            from argos.models.push import PushSubscription

            rows = (await s.execute(select(PushSubscription))).scalars().all()
            assert len(rows) == 1
            await s.refresh(rows[0])
            assert rows[0].p256dh == "k2"
            assert await remove_subscription(s, endpoint=ep) is True
            assert await remove_subscription(s, endpoint=ep) is False
