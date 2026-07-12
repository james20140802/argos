"""Unit + integration tests for argos.web.services.feed (ARG-155)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from urllib.parse import urlsplit

import pytest

from argos.config import settings
from argos.web.services.feed import (
    count_new_since,
    decode_cursor,
    decode_score_cursor,
    encode_cursor,
    encode_score_cursor,
    fetch_feed,
    select_hero,
    PAGE_SIZE,
    _reorder_diverse,
)
from tests.conftest import db_reachable as _db_reachable


# --------------------------------------------------------------------- #
# Pure cursor helpers (no DB required)
# --------------------------------------------------------------------- #


def test_page_size_default_is_20() -> None:
    assert PAGE_SIZE == 20


def test_encode_decode_cursor_round_trips() -> None:
    sort_at = datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc)
    item_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    token = encode_cursor(sort_at, item_id)
    parsed_sort, parsed_id = decode_cursor(token)

    assert parsed_sort == sort_at
    assert parsed_id == item_id


def test_encode_cursor_is_opaque_base64_string() -> None:
    sort_at = datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc)
    item_id = uuid.uuid4()
    token = encode_cursor(sort_at, item_id)

    assert isinstance(token, str)
    # No raw timestamp / UUID in token text.
    assert "2026-06-14" not in token
    assert str(item_id) not in token


def test_decode_cursor_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        decode_cursor("not-a-valid-cursor")


# --------------------------------------------------------------------- #
# DB-backed tests (self-skip on Release CI which has no Postgres)
# --------------------------------------------------------------------- #

_DB_URL: str = settings.database_url


pytestmark_db = pytest.mark.skipif(
    not _db_reachable(_DB_URL),
    reason="pgvector DB not reachable — skipping ARG-155 DB-backed tests",
)


def _utc(dt_iso: str) -> datetime:
    return datetime.fromisoformat(dt_iso).replace(tzinfo=timezone.utc)


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_feed_orders_newest_first_and_paginates_with_cursor() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    ids: list[uuid.UUID] = []
    try:
        # Seed 5 tech items spread out in published_at.
        async with Session() as session:
            base_t = datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)
            for i in range(5):
                item = TechItem(
                    title=f"feed-test-{i}",
                    source_url=f"https://example.com/arg155/{uuid.uuid4()}",
                    raw_content="x",
                    image_url=None,
                    category=CategoryType.MAINSTREAM,
                    trust_score=0.5,
                    published_at=base_t + timedelta(hours=i),
                )
                session.add(item)
                await session.flush()
                ids.append(item.id)
            await session.commit()
            seeded_ids = set(ids)

        async with Session() as session:
            # ARG-213 flips the *default* sort to "recommended" — this test
            # is specifically about the time-ordered path, so it must now say
            # so explicitly.
            page1 = await fetch_feed(session, limit=3, sort="latest")
            assert [it.id for it in page1.items if it.id in seeded_ids][:3] == list(reversed(ids))[:3]
            assert page1.next_cursor is not None

            page2 = await fetch_feed(session, cursor=page1.next_cursor, limit=3, sort="latest")
            page2_seeded = [it.id for it in page2.items if it.id in seeded_ids]
            assert list(reversed(ids))[3:5] == page2_seeded[:2]
    finally:
        # Cleanup so reruns stay clean — in `finally` (not after the asserts)
        # so a failed assertion never leaks seeded rows (ARG-191).
        if ids:
            async with Session() as session:
                for tid in ids:
                    obj = await session.get(TechItem, tid)
                    if obj is not None:
                        await session.delete(obj)
                await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_feed_filters_by_category_and_joins_status() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem
    from argos.models.user_asset import AssetStatus, UserAsset  # noqa: F401 — used in asset creation

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            mainstream = TechItem(
                title="arg155-mainstream",
                source_url=f"https://example.com/arg155/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                published_at=datetime(2026, 6, 14, 1, 0, tzinfo=timezone.utc),
            )
            alpha = TechItem(
                title="arg155-alpha",
                source_url=f"https://example.com/arg155/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                published_at=datetime(2026, 6, 14, 2, 0, tzinfo=timezone.utc),
            )
            session.add_all([mainstream, alpha])
            await session.flush()
            seeded_ids = [mainstream.id, alpha.id]
            asset = UserAsset(tech_id=alpha.id, status=AssetStatus.KEEP)
            session.add(asset)
            await session.commit()

        async with Session() as session:
            page = await fetch_feed(session, category="Alpha", limit=20)
            ours = [it for it in page.items if it.id in set(seeded_ids)]
            assert all(it.category == CategoryType.ALPHA for it in ours)
            kept = [it for it in ours if it.id == alpha.id]
            assert len(kept) == 1 and kept[0].status == AssetStatus.KEEP

            mainstream_page = await fetch_feed(session, category="Mainstream", limit=20)
            ours_main = [it for it in mainstream_page.items if it.id in set(seeded_ids)]
            assert all(it.category == CategoryType.MAINSTREAM for it in ours_main)
    finally:
        # Core DELETE so the DB-level FK CASCADE removes the seeded
        # user_assets row. An ORM session.delete(TechItem) would instead
        # try to NULL the child's tech_id (no delete-orphan on the
        # relationship), violating its NOT NULL constraint.
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_feed_returns_summary_and_none_when_null() -> None:
    """ARG-174 (T1): fetch_feed must project TechItem.summary onto FeedItem,
    surfacing the triage-generated one-liner (or None when the column is
    null) so downstream cards can render it."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            with_summary = TechItem(
                title="arg174-with-summary",
                source_url=f"https://example.com/arg174/{uuid.uuid4()}",
                raw_content="x",
                summary="한 줄 요약입니다.",
                category=CategoryType.MAINSTREAM,
                # Far-future published_at so both seeded rows land on the first
                # page regardless of how many real items a populated dev DB has
                # accumulated — otherwise they fall off page 1 and the lookup
                # below raises an opaque KeyError instead of asserting summary.
                published_at=datetime(2099, 1, 1, 2, 0, tzinfo=timezone.utc),
            )
            without_summary = TechItem(
                title="arg174-no-summary",
                source_url=f"https://example.com/arg174/{uuid.uuid4()}",
                raw_content="x",
                summary=None,
                category=CategoryType.MAINSTREAM,
                published_at=datetime(2099, 1, 1, 1, 0, tzinfo=timezone.utc),
            )
            session.add_all([with_summary, without_summary])
            await session.flush()
            seeded_ids = [with_summary.id, without_summary.id]
            await session.commit()

        async with Session() as session:
            # published_at=2099 only guarantees first-page placement under the
            # time-ordered path — pin sort="latest" explicitly (ARG-213 made
            # "recommended"/feed_score the default).
            page = await fetch_feed(session, limit=PAGE_SIZE, sort="latest")
            by_id = {it.id: it for it in page.items if it.id in set(seeded_ids)}
            assert set(seeded_ids) <= by_id.keys(), (
                "seeded items must appear on the first page"
            )
            assert by_id[seeded_ids[0]].summary == "한 줄 요약입니다."
            assert by_id[seeded_ids[1]].summary is None
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_count_new_since_counts_only_items_after_cursor() -> None:
    """ARG-203: count_new_since must count items sorting strictly after the
    cursor position (or tied on sort_at with a greater id), matching
    fetch_feed's ordering rule inverted."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    # Far-future timestamps so this trio is unambiguously ordered relative to
    # any pre-existing dev-DB rows (mirrors the ARG-174 test's technique).
    base_t = datetime(2099, 6, 1, 0, 0, tzinfo=timezone.utc)
    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            older = TechItem(
                title="arg203-older",
                source_url=f"https://example.com/arg203/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                published_at=base_t,
            )
            newer1 = TechItem(
                title="arg203-newer1",
                source_url=f"https://example.com/arg203/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                published_at=base_t + timedelta(hours=1),
            )
            newer2 = TechItem(
                title="arg203-newer2",
                source_url=f"https://example.com/arg203/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                published_at=base_t + timedelta(hours=2),
            )
            session.add_all([older, newer1, newer2])
            await session.flush()
            seeded_ids = [older.id, newer1.id, newer2.id]
            await session.commit()

        cursor = encode_cursor(older.published_at, older.id)

        async with Session() as session:
            n = await count_new_since(session, cursor=cursor)
            assert n >= 2  # at least the two seeded newer rows

            n_alpha = await count_new_since(session, category="Alpha", cursor=cursor)
            assert n_alpha >= 1

        with pytest.raises(ValueError):
            async with Session() as session:
                await count_new_since(session, cursor="garbage")
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


def test_fetch_feed_module_does_not_reference_briefed_at() -> None:
    """Slack briefing owns ``briefed_at``; this service must not touch it."""
    import inspect

    from argos.web.services import feed as feed_mod

    src = inspect.getsource(feed_mod)
    assert "briefed_at" not in src, (
        "argos.web.services.feed must not reference briefed_at "
        "(Slack briefing owns that column)"
    )


# --------------------------------------------------------------------- #
# ARG-213 — recommended sort, tagged cursors, domain diversity, hero
# --------------------------------------------------------------------- #


def test_encode_decode_score_cursor_round_trips() -> None:
    item_id = uuid.uuid4()
    token = encode_score_cursor(0.42, item_id)
    score, parsed_id = decode_score_cursor(token)
    assert score == pytest.approx(0.42)
    assert parsed_id == item_id


def test_encode_decode_score_cursor_round_trips_with_none_score() -> None:
    """A NULLS-LAST boundary row has ``feed_score is None`` — the cursor must
    round-trip that faithfully rather than coercing it to 0.0 or erroring."""
    item_id = uuid.uuid4()
    token = encode_score_cursor(None, item_id)
    score, parsed_id = decode_score_cursor(token)
    assert score is None
    assert parsed_id == item_id


def test_decode_score_cursor_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        decode_score_cursor("not-a-valid-cursor")


def test_decode_score_cursor_rejects_latest_tagged_cursor() -> None:
    """A ``latest``-sort cursor fed into the ``recommended`` path must raise
    — not silently reinterpret a timestamp payload as a feed_score (ARG-213
    AC: mixing sort cursors -> ValueError -> 400)."""
    latest_cursor = encode_cursor(
        datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc), uuid.uuid4()
    )
    with pytest.raises(ValueError):
        decode_score_cursor(latest_cursor)


def test_decode_cursor_rejects_score_tagged_cursor() -> None:
    """The inverse direction of the same AC: a ``recommended``-sort cursor
    fed into the ``latest`` path must also raise."""
    score_cursor = encode_score_cursor(0.9, uuid.uuid4())
    with pytest.raises(ValueError):
        decode_cursor(score_cursor)


@pytest.mark.asyncio
async def test_fetch_feed_rejects_cross_sort_cursor_score_into_latest() -> None:
    """Cursor tag validation happens before any DB access, so this is
    provable without a live session (``None`` is never touched)."""
    bad = encode_score_cursor(0.5, uuid.uuid4())
    with pytest.raises(ValueError):
        await fetch_feed(None, cursor=bad, sort="latest")


@pytest.mark.asyncio
async def test_fetch_feed_rejects_cross_sort_cursor_latest_into_recommended() -> None:
    bad = encode_cursor(datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc), uuid.uuid4())
    with pytest.raises(ValueError):
        await fetch_feed(None, cursor=bad, sort="recommended")


# ---- _reorder_diverse: pure function, no DB ---- #


def test_reorder_diverse_breaks_runs():
    class _Item:
        def __init__(self, u):
            self.source_url = u

    items = [_Item("https://a.com/1"), _Item("https://a.com/2"), _Item("https://b.com/1")]
    out = _reorder_diverse(items)
    domains = [i.source_url.split("/")[2] for i in out]
    assert not (domains[0] == domains[1])  # 연속 방지
    assert len(out) == 3  # 손실 없음


def test_reorder_diverse_keeps_original_order_when_unavoidable():
    """All items share one domain — nothing can be un-consecutive, so the
    original relative order must be preserved rather than shuffled."""
    class _Item:
        def __init__(self, u):
            self.source_url = u

    items = [_Item("https://a.com/1"), _Item("https://a.com/2"), _Item("https://a.com/3")]
    out = _reorder_diverse(items)
    assert [i.source_url for i in out] == [i.source_url for i in items]


def test_reorder_diverse_handles_empty_and_singleton():
    assert _reorder_diverse([]) == []

    class _Item:
        def __init__(self, u):
            self.source_url = u

    solo = [_Item("https://a.com/1")]
    assert [i.source_url for i in _reorder_diverse(solo)] == ["https://a.com/1"]


def test_reorder_diverse_avoids_avoidable_run_on_skewed_page():
    """Regression: a naive "just differ from the immediately preceding item"
    greedy can still leave an avoidable same-domain run near the end of a
    domain-skewed page — e.g. exactly half the page from one domain is fully
    alternate-able, but draining the smaller domains first (because they sit
    earlier in the page) can strand the dominant domain's items at the tail.
    10 dominant + 10 spread across 5 other domains (matching a real feed
    shape) must reorder with zero adjacent same-domain pairs."""

    class _Item:
        def __init__(self, u):
            self.source_url = u

    # Deliberately front-load the minority domains and push the dominant
    # domain toward the end, mirroring the failure mode.
    minority = (
        ["https://huggingface.co/x"] * 3
        + ["https://example.com/x"] * 4
        + ["https://zerofs.net/x"]
        + ["https://cornell.edu/x"]
        + ["https://blog.kog.ai/x"]
    )
    items = [_Item(u) for u in minority] + [_Item("https://openai.com/x")] * 10
    out = _reorder_diverse(items)

    assert len(out) == len(items)
    domains = [urlsplit(i.source_url).netloc for i in out]
    for a, b in zip(domains, domains[1:]):
        assert a != b, f"avoidable adjacent same-domain pair: {domains}"


# --------------------------------------------------------------------- #
# DB-backed: recommended-sort ordering / pagination / hero
# --------------------------------------------------------------------- #


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_feed_recommended_sort_orders_by_feed_score_nulls_last() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            high = TechItem(
                title="arg213-high",
                source_url=f"https://example.com/arg213/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                feed_score=900.0,
            )
            mid = TechItem(
                title="arg213-mid",
                source_url=f"https://example.com/arg213/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                feed_score=500.0,
            )
            low = TechItem(
                title="arg213-low",
                source_url=f"https://example.com/arg213/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                feed_score=100.0,
            )
            null_score = TechItem(
                title="arg213-null",
                source_url=f"https://example.com/arg213/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.MAINSTREAM,
                feed_score=None,
            )
            session.add_all([high, mid, low, null_score])
            await session.flush()
            seeded_ids = [high.id, mid.id, low.id, null_score.id]
            await session.commit()

        async with Session() as session:
            page = await fetch_feed(session, sort="recommended", limit=200)
            ours = [it for it in page.items if it.id in set(seeded_ids)]
            ours_by_id = {it.id: it for it in ours}
            assert set(seeded_ids) == ours_by_id.keys()
            # feed_score descending, NULL last: each named item's position in
            # the returned page must be non-decreasing across this sequence.
            positions = [
                ours.index(ours_by_id[i])
                for i in [high.id, mid.id, low.id, null_score.id]
            ]
            assert positions == sorted(positions)
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_feed_recommended_sort_cursor_pagination_no_dupes_or_gaps() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            items = []
            # A distinct, non-overlapping score band per item plus a NULL tail
            # item, all under one throwaway category filter so pagination can
            # be verified against a known, exact set (page-1 + page-2 must
            # union back to exactly these 6, no dupes/gaps).
            for i, score in enumerate([600.0, 500.0, 400.0, 300.0, 200.0, None]):
                item = TechItem(
                    title=f"arg213-page-{i}",
                    source_url=f"https://example.com/arg213page/{uuid.uuid4()}",
                    raw_content="x",
                    category=CategoryType.ALPHA,
                    feed_score=score,
                )
                items.append(item)
            session.add_all(items)
            await session.flush()
            seeded_ids = [it.id for it in items]
            await session.commit()

        async with Session() as session:
            page1 = await fetch_feed(
                session, category="Alpha", sort="recommended", limit=3
            )
            page1_ids = [it.id for it in page1.items if it.id in set(seeded_ids)]
            assert page1.next_cursor is not None

            page2 = await fetch_feed(
                session,
                category="Alpha",
                sort="recommended",
                cursor=page1.next_cursor,
                limit=200,
            )
            page2_ids = [it.id for it in page2.items if it.id in set(seeded_ids)]

            all_ids = page1_ids + page2_ids
            assert len(all_ids) == len(set(all_ids)) == len(seeded_ids), (
                "no duplicates or gaps across the cursor boundary"
            )
            assert set(all_ids) == set(seeded_ids)
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_feed_latest_sort_cursor_pagination_no_dupes_or_gaps() -> None:
    """Mirrors the recommended-sort pagination test above for the explicit
    ``sort="latest"`` path — both sorts must round-trip cleanly."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    base_t = datetime(2026, 6, 20, 0, 0, tzinfo=timezone.utc)
    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            items = [
                TechItem(
                    title=f"arg213-latest-{i}",
                    source_url=f"https://example.com/arg213latest/{uuid.uuid4()}",
                    raw_content="x",
                    category=CategoryType.ALPHA,
                    published_at=base_t + timedelta(hours=i),
                )
                for i in range(6)
            ]
            session.add_all(items)
            await session.flush()
            seeded_ids = [it.id for it in items]
            await session.commit()

        async with Session() as session:
            page1 = await fetch_feed(
                session, category="Alpha", sort="latest", limit=3
            )
            page1_ids = [it.id for it in page1.items if it.id in set(seeded_ids)]
            assert page1.next_cursor is not None

            page2 = await fetch_feed(
                session,
                category="Alpha",
                sort="latest",
                cursor=page1.next_cursor,
                limit=200,
            )
            page2_ids = [it.id for it in page2.items if it.id in set(seeded_ids)]

            all_ids = page1_ids + page2_ids
            assert len(all_ids) == len(set(all_ids)) == len(seeded_ids)
            assert set(all_ids) == set(seeded_ids)
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_select_hero_prefers_highest_score_within_48h() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            recent_high = TechItem(
                title="arg213-hero-recent-high",
                source_url=f"https://example.com/arg213hero/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                feed_score=999.0,
                created_at=now - timedelta(hours=2),
            )
            recent_low = TechItem(
                title="arg213-hero-recent-low",
                source_url=f"https://example.com/arg213hero/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                feed_score=100.0,
                created_at=now - timedelta(hours=1),
            )
            old_higher = TechItem(
                title="arg213-hero-old-higher",
                source_url=f"https://example.com/arg213hero/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                feed_score=5000.0,
                created_at=now - timedelta(hours=72),
            )
            session.add_all([recent_high, recent_low, old_higher])
            await session.flush()
            seeded_ids = [recent_high.id, recent_low.id, old_higher.id]
            await session.commit()

        async with Session() as session:
            hero_id = await select_hero(session, category="Alpha")
            assert hero_id == recent_high.id, (
                "the higher-scored item outside the 48h window must lose to "
                "the highest-scored item within it"
            )
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_select_hero_falls_back_to_global_highest_when_none_within_48h() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    now = datetime.now(timezone.utc)
    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            old_a = TechItem(
                title="arg213-hero-fallback-a",
                source_url=f"https://example.com/arg213heroFB/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                feed_score=300.0,
                created_at=now - timedelta(hours=96),
            )
            old_b = TechItem(
                title="arg213-hero-fallback-b",
                source_url=f"https://example.com/arg213heroFB/{uuid.uuid4()}",
                raw_content="x",
                category=CategoryType.ALPHA,
                feed_score=700.0,
                created_at=now - timedelta(hours=120),
            )
            session.add_all([old_a, old_b])
            await session.flush()
            seeded_ids = [old_a.id, old_b.id]
            await session.commit()

        async with Session() as session:
            hero_id = await select_hero(session, category="Alpha")
            assert hero_id == old_b.id, (
                "with nothing inside the 48h window, fall back to the "
                "highest feed_score overall"
            )
    finally:
        from sqlalchemy import delete as sa_delete

        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_select_hero_returns_none_when_nothing_scored() -> None:
    """Deterministic empty-state check: snapshot + clear every feed_score,
    assert None, then restore — independent of whatever other tests in this
    session left behind."""
    from sqlalchemy import text as sa_text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)

    try:
        async with Session() as session:
            snapshot = (
                await session.execute(
                    sa_text(
                        "SELECT id, feed_score FROM tech_items "
                        "WHERE feed_score IS NOT NULL"
                    )
                )
            ).all()
            await session.execute(
                sa_text("UPDATE tech_items SET feed_score = NULL")
            )
            await session.commit()

        try:
            async with Session() as session:
                assert await select_hero(session) is None
                assert await select_hero(session, category="Alpha") is None
        finally:
            async with Session() as session:
                for row in snapshot:
                    await session.execute(
                        sa_text(
                            "UPDATE tech_items SET feed_score = :s WHERE id = :i"
                        ),
                        {"s": row.feed_score, "i": row.id},
                    )
                await session.commit()
    finally:
        await engine.dispose()
