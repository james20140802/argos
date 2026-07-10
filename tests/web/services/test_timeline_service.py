"""Unit + integration tests for argos.web.services.timeline (ARG-205)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from argos.config import settings
from tests.conftest import db_reachable as _db_reachable


_DB_URL: str = settings.database_url


pytestmark_db = pytest.mark.skipif(
    not _db_reachable(_DB_URL),
    reason="pgvector DB not reachable — skipping ARG-205 DB-backed tests",
)


def test_timeline_event_is_frozen_dataclass() -> None:
    from argos.web.services.timeline import TimelineEvent

    event = TimelineEvent(
        kind="status",
        changed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        title=None,
        link_tech_id=None,
        changed_from="Tracking",
        changed_to="Keep",
        relation_type=None,
        reasoning=None,
        label="Tracking → Keep",
    )
    with pytest.raises(Exception):
        event.label = "mutated"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_fetch_timeline_returns_empty_for_empty_asset() -> None:
    """An asset with no track_history/succession rows returns []."""
    from argos.web.services.timeline import fetch_timeline

    class _Sentinel:
        async def execute(self, *args, **kwargs):
            class _Result:
                def all(self_inner):
                    return []

            return _Result()

    result = await fetch_timeline(_Sentinel(), uuid.uuid4())  # type: ignore[arg-type]
    assert result == []


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_timeline_merges_and_sorts_reverse_chronological() -> None:
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.tech_succession import RelationType, TechSuccession
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.timeline import fetch_timeline

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_tech_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            # P: the Keep asset whose timeline we're building.
            p = TechItem(
                title="arg205-p",
                source_url=f"https://example.com/arg205/{uuid.uuid4()}",
                raw_content="x",
            )
            # Q: the item matched by a SIGNAL_MATCHED alert on P.
            q = TechItem(
                title="arg205-q",
                source_url=f"https://example.com/arg205/{uuid.uuid4()}",
                raw_content="x",
            )
            # S: the successor of P via tech_succession.
            s = TechItem(
                title="arg205-s",
                source_url=f"https://example.com/arg205/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add_all([p, q, s])
            await session.flush()
            seeded_tech_ids = [p.id, q.id, s.id]

            p_asset = UserAsset(tech_id=p.id, status=AssetStatus.KEEP)
            session.add(p_asset)
            await session.flush()

            session.add(
                TrackHistory(
                    user_asset_id=p_asset.id,
                    changed_from="Tracking",
                    changed_to="Keep",
                    changed_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                )
            )
            session.add(
                TrackHistory(
                    user_asset_id=p_asset.id,
                    changed_from=str(q.id),
                    changed_to="signal_matched",
                    changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                )
            )
            session.add(
                TechSuccession(
                    predecessor_id=p.id,
                    successor_id=s.id,
                    relation_type=RelationType.ENHANCE,
                    reasoning="s enhances p",
                    created_at=datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc),
                )
            )
            await session.commit()
            p_id = p.id
            q_id = q.id
            s_id = s.id

        async with Session() as session:
            events = await fetch_timeline(session, p_id)
            assert [e.kind for e in events] == ["succession", "signal", "status"]
            assert any(
                e.kind == "signal" and e.link_tech_id == q_id and e.title == "arg205-q"
                for e in events
            )
            succ = next(e for e in events if e.kind == "succession")
            assert succ.link_tech_id == s_id
            assert succ.relation_type == RelationType.ENHANCE

            # limit
            limited = await fetch_timeline(session, p_id, limit=1)
            assert len(limited) == 1
            assert limited[0].kind == "succession"

            # empty asset
            assert await fetch_timeline(session, uuid.uuid4()) == []
    finally:
        async with Session() as session:
            if seeded_tech_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_tech_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_timeline_excludes_deleted_signal_match_target() -> None:
    """A SIGNAL_MATCHED row whose matched item was deleted is silently
    excluded (inner-join effect) rather than surfacing with a None title."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.timeline import fetch_timeline

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_tech_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            p = TechItem(
                title="arg205-deleted-match-p",
                source_url=f"https://example.com/arg205/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add(p)
            await session.flush()
            seeded_tech_ids = [p.id]
            p_asset = UserAsset(tech_id=p.id, status=AssetStatus.KEEP)
            session.add(p_asset)
            await session.flush()
            # changed_from points at a UUID with no matching tech_item row.
            session.add(
                TrackHistory(
                    user_asset_id=p_asset.id,
                    changed_from=str(uuid.uuid4()),
                    changed_to="signal_matched",
                    changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                )
            )
            await session.commit()
            p_id = p.id

        async with Session() as session:
            events = await fetch_timeline(session, p_id)
            assert events == []
    finally:
        async with Session() as session:
            if seeded_tech_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_tech_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_timeline_legacy_succession_alerted_is_plain_text_event() -> None:
    """A legacy succession_alerted row with changed_from='Keep' can't be
    resolved to a specific successor — it surfaces as a title=None event."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.timeline import fetch_timeline

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_tech_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            p = TechItem(
                title="arg205-legacy-succ",
                source_url=f"https://example.com/arg205/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add(p)
            await session.flush()
            seeded_tech_ids = [p.id]
            p_asset = UserAsset(tech_id=p.id, status=AssetStatus.KEEP)
            session.add(p_asset)
            await session.flush()
            session.add(
                TrackHistory(
                    user_asset_id=p_asset.id,
                    changed_from="Keep",
                    changed_to="succession_alerted",
                    changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                )
            )
            await session.commit()
            p_id = p.id

        async with Session() as session:
            events = await fetch_timeline(session, p_id)
            assert len(events) == 1
            assert events[0].kind == "signal"
            assert events[0].title is None
            assert events[0].link_tech_id is None
            assert events[0].label == "후속 기술 신호"
    finally:
        async with Session() as session:
            if seeded_tech_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_tech_ids))
                )
            await session.commit()
        await engine.dispose()
