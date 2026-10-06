"""ARG-243: the feed's unit is the event — DB-backed entry semantics.

Self-skips without Postgres (release CI), like the rest of the feed tests.
Each test seeds its own rows under a unique URL prefix and tears them down in
``finally`` so a failed assertion never leaks rows into the scratch DB.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from argos.config import settings
from argos.web.services.feed import (
    count_new_since,
    encode_cursor,
    fetch_feed,
    fetch_feed_entry,
    latest_feed_cursor,
    select_hero,
)
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url

pytestmark = [
    pytest.mark.skipif(
        not _db_reachable(_DB_URL),
        reason="pgvector DB not reachable — skipping ARG-243 DB-backed tests",
    ),
    pytest.mark.asyncio,
]

# Far in the future so seeded rows always lead the "latest" order regardless of
# whatever else lives in the scratch DB.
BASE = datetime(2099, 1, 1, tzinfo=timezone.utc)


@asynccontextmanager
async def _seeded():
    """Yield ``(Session, seed)``; everything ``seed`` creates is removed after."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.event_document import EventDocument
    from argos.models.tech_event import TechEvent
    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    created_items: list[uuid.UUID] = []
    created_events: list[uuid.UUID] = []

    class Seed:
        async def item(
            self,
            session,
            *,
            domain: str,
            hours: float,
            score: float | None = None,
            category=CategoryType.MAINSTREAM,
            image: str | None = None,
        ):
            row = TechItem(
                title=f"doc {domain} {hours}",
                source_url=f"https://{domain}/arg243/{uuid.uuid4()}",
                raw_content="x",
                image_url=image,
                category=category,
                published_at=BASE + timedelta(hours=hours),
                feed_score=score,
            )
            session.add(row)
            await session.flush()
            created_items.append(row.id)
            return row

        async def event(self, session, docs, *, title="사건", merged_into=None):
            ev = TechEvent(
                title=title,
                summary="사건 요약",
                occurred_at=BASE,
                merged_into_id=merged_into,
            )
            session.add(ev)
            await session.flush()
            created_events.append(ev.id)
            for d in docs:
                session.add(EventDocument(event_id=ev.id, tech_item_id=d.id))
            await session.flush()
            return ev

    try:
        yield Session, Seed()
    finally:
        async with Session() as session:
            # Tombstones first: merged_into_id is ON DELETE RESTRICT.
            events =[await session.get(TechEvent, eid) for eid in created_events]
            for ev in sorted(
                (e for e in events if e is not None),
                key=lambda e: e.merged_into_id is None,
            ):
                await session.delete(ev)
                await session.flush()
            for tid in created_items:
                obj = await session.get(TechItem, tid)
                if obj is not None:
                    await session.delete(obj)
            await session.commit()
        await engine.dispose()


def _mine(items, ids):
    return [it for it in items if it.id in ids]


async def test_event_collapses_its_documents_into_one_entry() -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            first = await seed.item(s, domain="a.example", hours=1, score=0.2)
            mid = await seed.item(s, domain="b.example", hours=2, score=0.9)
            last = await seed.item(s, domain="www.c.example", hours=3, score=0.4)
            again = await seed.item(s, domain="a.example", hours=4, score=None)
            ev = await seed.event(s, [mid, last, first, again], title="Sonnet 5 출시")
            await s.commit()

        async with Session() as s:
            page = await fetch_feed(s, sort="latest", limit=50)
        doc_ids = {first.id, mid.id, last.id, again.id}
        assert not _mine(page.items, doc_ids), "evidence docs must not get their own cards"
        [entry] = _mine(page.items, {ev.id})
        assert entry.kind == "event"
        assert entry.title == "Sonnet 5 출시"
        assert entry.rep_id == first.id  # earliest report is the representative
        assert entry.action_id == first.id
        assert entry.sort_at == BASE + timedelta(hours=4)  # newest evidence
        assert entry.feed_score == pytest.approx(0.9)  # max evidence score
        assert entry.doc_count == 4
        assert entry.source_domains == ("a.example", "b.example", "c.example")
        assert entry.source_count == 3
        assert entry.href == f"/event/{ev.id}"


async def test_lone_document_is_a_one_source_entry_on_its_old_url() -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            lone = await seed.item(s, domain="solo.example", hours=5)
            await s.commit()
        async with Session() as s:
            page = await fetch_feed(s, sort="latest", limit=50)
        [entry] = _mine(page.items, {lone.id})
        assert entry.kind == "item"
        assert entry.rep_id == lone.id
        assert entry.source_count == 1
        assert entry.href == f"/item/{lone.id}"


async def test_unnamed_event_borrows_representative_title() -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            d = await seed.item(s, domain="a.example", hours=1)
            ev = await seed.event(s, [d], title=None)
            await s.commit()
        async with Session() as s:
            entry = await fetch_feed_entry(s, ev.id)
        assert entry is not None and entry.title == d.title


async def test_tombstoned_event_hides_but_its_documents_stay_visible() -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            keep_doc = await seed.item(s, domain="a.example", hours=1)
            orphan = await seed.item(s, domain="b.example", hours=2)
            survivor = await seed.event(s, [keep_doc])
            absorbed = await seed.event(s, [orphan], merged_into=survivor.id)
            await s.commit()
        async with Session() as s:
            page = await fetch_feed(s, sort="latest", limit=50)
        ids = {it.id for it in page.items}
        assert absorbed.id not in ids
        assert survivor.id in ids
        # Linked only to a tombstone: shown alone rather than silently dropped.
        assert orphan.id in ids


async def test_document_in_two_events_backs_both_cards() -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            shared = await seed.item(s, domain="a.example", hours=1)
            other = await seed.item(s, domain="b.example", hours=2)
            e1 = await seed.event(s, [shared])
            e2 = await seed.event(s, [shared, other])
            await s.commit()
        async with Session() as s:
            page = await fetch_feed(s, sort="latest", limit=50)
        assert {it.id for it in _mine(page.items, {e1.id, e2.id})} == {e1.id, e2.id}
        assert shared.id not in {it.id for it in page.items}


async def test_category_filter_follows_representative() -> None:
    from argos.models.tech_item import CategoryType

    async with _seeded() as (Session, seed):
        async with Session() as s:
            rep = await seed.item(s, domain="a.example", hours=1, category=CategoryType.ALPHA)
            later = await seed.item(s, domain="b.example", hours=2)
            ev = await seed.event(s, [rep, later])
            await s.commit()
        async with Session() as s:
            alpha = await fetch_feed(s, sort="latest", category="Alpha", limit=50)
            main = await fetch_feed(s, sort="latest", category="Mainstream", limit=50)
        assert ev.id in {it.id for it in alpha.items}
        assert ev.id not in {it.id for it in main.items}


@pytest.mark.parametrize("sort", ["latest", "recommended"])
async def test_pagination_over_mixed_entries_has_no_dupes_or_gaps(sort) -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            expected = set()
            for i in range(4):
                docs = [
                    await seed.item(s, domain=f"e{i}-{j}.example", hours=10 + i * 3 + j,
                                    score=(0.5 + i / 10) if j == 0 else None)
                    for j in range(2)
                ]
                expected.add((await seed.event(s, docs)).id)
            for i in range(5):
                expected.add(
                    (await seed.item(s, domain=f"lone{i}.example", hours=40 + i,
                                     score=0.55 if i % 2 else None)).id
                )
            await s.commit()

        seen: list[uuid.UUID] = []
        cursor = None
        async with Session() as s:
            for _ in range(50):
                page = await fetch_feed(s, sort=sort, cursor=cursor, limit=3)
                seen.extend(it.id for it in page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
        mine = [i for i in seen if i in expected]
        assert len(mine) == len(set(mine)), "an entry appeared on two pages"
        assert set(mine) == expected, "an entry was skipped between pages"


async def test_new_evidence_on_existing_event_counts_as_new() -> None:
    async with _seeded() as (Session, seed):
        async with Session() as s:
            d1 = await seed.item(s, domain="a.example", hours=100)
            ev = await seed.event(s, [d1])
            await s.commit()
        async with Session() as s:
            baseline = await latest_feed_cursor(s)
        assert baseline == encode_cursor(BASE + timedelta(hours=100), ev.id)

        from argos.models.event_document import EventDocument

        async with Session() as s:
            d2 = await seed.item(s, domain="b.example", hours=101)
            s.add(EventDocument(event_id=ev.id, tech_item_id=d2.id))
            await s.commit()
        async with Session() as s:
            assert await count_new_since(s, cursor=baseline) == 1


async def test_select_hero_returns_entry_id() -> None:
    now = datetime.now(timezone.utc)
    async with _seeded() as (Session, seed):
        async with Session() as s:
            hours = (now - BASE).total_seconds() / 3600 - 1  # one hour ago
            d1 = await seed.item(s, domain="a.example", hours=hours - 2, score=0.1)
            d2 = await seed.item(s, domain="b.example", hours=hours, score=9.99)
            ev = await seed.event(s, [d1, d2])
            await s.commit()
        async with Session() as s:
            assert await select_hero(s) == ev.id


async def test_event_detail_lists_evidence_in_report_order() -> None:
    from argos.web.services.event_detail import fetch_event_detail

    async with _seeded() as (Session, seed):
        async with Session() as s:
            late = await seed.item(s, domain="late.example", hours=9)
            first = await seed.item(s, domain="www.first.example", hours=1, image="https://img/x.png")
            mid = await seed.item(s, domain="mid.example", hours=5)
            ev = await seed.event(s, [late, first, mid], title="사건 A")
            await s.commit()
        async with Session() as s:
            view = await fetch_event_detail(s, ev.id)
            missing = await fetch_event_detail(s, uuid.uuid4())
    assert missing is None
    assert [d.id for d in view.documents] == [first.id, mid.id, late.id]
    assert [d.is_first for d in view.documents] == [True, False, False]
    assert view.documents[0].domain == "first.example"
    assert view.rep_id == first.id
    assert view.image_url == "https://img/x.png"
    assert view.source_count == 3
