"""Unit + integration tests for argos.web.services.feed (ARG-155)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from argos.config import settings
from argos.web.services.feed import (
    decode_cursor,
    encode_cursor,
    fetch_feed,
    PAGE_SIZE,
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
            page1 = await fetch_feed(session, limit=3)
            assert [it.id for it in page1.items if it.id in seeded_ids][:3] == list(reversed(ids))[:3]
            assert page1.next_cursor is not None

            page2 = await fetch_feed(session, cursor=page1.next_cursor, limit=3)
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
            page = await fetch_feed(session, limit=PAGE_SIZE)
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


def test_fetch_feed_module_does_not_reference_briefed_at() -> None:
    """Slack briefing owns ``briefed_at``; this service must not touch it."""
    import inspect

    from argos.web.services import feed as feed_mod

    src = inspect.getsource(feed_mod)
    assert "briefed_at" not in src, (
        "argos.web.services.feed must not reference briefed_at "
        "(Slack briefing owns that column)"
    )
