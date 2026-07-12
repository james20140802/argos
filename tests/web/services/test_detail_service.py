"""Unit + integration tests for argos.web.services.detail (ARG-158)."""
from __future__ import annotations

import uuid

import pytest

from argos.config import settings
from argos.web.services.detail import ItemDetailView, fetch_item_detail
from tests.conftest import db_reachable as _db_reachable


_DB_URL: str = settings.database_url


pytestmark_db = pytest.mark.skipif(
    not _db_reachable(_DB_URL),
    reason="pgvector DB not reachable — skipping ARG-158 DB-backed tests",
)


def test_item_detail_view_is_frozen_dataclass() -> None:
    view = ItemDetailView(
        id=uuid.uuid4(),
        title="t",
        source_url="https://example.com",
        image_url=None,
        summary=None,
        category=None,
        trust_score=None,
        published_at=None,
    )
    with pytest.raises(Exception):
        view.title = "mutated"  # type: ignore[misc]


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_item_detail_returns_none_for_unknown_id() -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with Session() as session:
            view = await fetch_item_detail(session, uuid.uuid4())
            assert view is None
    finally:
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_item_detail_returns_view_for_known_id() -> None:
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_id: uuid.UUID | None = None
    try:
        async with Session() as session:
            item = TechItem(
                title="arg158-detail",
                source_url=f"https://example.com/arg158/{uuid.uuid4()}",
                raw_content="raw",
                summary="A long form summary that the reader will see.",
                category=CategoryType.ALPHA,
                trust_score=0.66,
            )
            session.add(item)
            await session.flush()
            seeded_id = item.id
            await session.commit()

        async with Session() as session:
            view = await fetch_item_detail(session, seeded_id)
            assert view is not None
            assert view.title == "arg158-detail"
            assert view.summary == "A long form summary that the reader will see."
            assert view.category == CategoryType.ALPHA
            assert view.trust_score == pytest.approx(0.66)
            assert view.predecessors == []
            assert view.successors == []
    finally:
        async with Session() as session:
            if seeded_id is not None:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id == seeded_id)
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_similar_ranks_keep_assets_by_proximity_to_current_item() -> None:
    """Direct check on the similarity helper.

    The comparison is anchored on the *current item*: candidates are the
    user's Keep assets, ranked by cosine distance to the viewed item. A
    non-Keep item close to the anchor must NOT surface, and a Keep asset's
    own self-match (when it is the current item) is excluded by id.

    Uses a generous LIMIT so the assertion ranks SEEDED Keep assets against
    each other — the shared dev DB has its own Keep assets and embedded
    items that would otherwise crowd the top-5 production ordering.
    """
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.detail import _fetch_similar

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            anchor_emb = [1.0] + [0.0] * 767
            very_similar = [0.99] + [0.01] * 767
            somewhat = [0.5] + [0.5] * 767
            # Far point on a perpendicular axis — cosine distance ≫ above.
            far = [0.0] * 767 + [1.0]

            # The item being viewed (not a Keep asset itself).
            anchor = TechItem(
                title="arg160-current",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=anchor_emb,
            )
            # Keep assets at varying distance from the anchor.
            keep_near = TechItem(
                title="arg160-keep-near",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=very_similar,
            )
            keep_mid = TechItem(
                title="arg160-keep-mid",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=somewhat,
            )
            keep_far = TechItem(
                title="arg160-keep-far",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=far,
            )
            keep_no_emb = TechItem(
                title="arg160-keep-no-emb",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=None,
            )
            # Close to the anchor but NOT a Keep asset — must never surface.
            nonkeep_near = TechItem(
                title="arg160-nonkeep-near",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=very_similar,
            )
            session.add_all(
                [anchor, keep_near, keep_mid, keep_far, keep_no_emb, nonkeep_near]
            )
            await session.flush()
            seeded_ids = [
                anchor.id,
                keep_near.id,
                keep_mid.id,
                keep_far.id,
                keep_no_emb.id,
                nonkeep_near.id,
            ]
            session.add_all(
                [
                    UserAsset(tech_id=keep_near.id, status=AssetStatus.KEEP),
                    UserAsset(tech_id=keep_mid.id, status=AssetStatus.KEEP),
                    UserAsset(tech_id=keep_far.id, status=AssetStatus.KEEP),
                    UserAsset(tech_id=keep_no_emb.id, status=AssetStatus.KEEP),
                ]
            )
            await session.commit()
            anchor_id = anchor.id

        async with Session() as session:
            # Generous limit so seeded titles all surface regardless of
            # production data populating the top-K.
            similar = await _fetch_similar(session, anchor_id, limit=5000)
            titles = [s.title for s in similar]

            # Current item, the no-embedding Keep asset, and any non-Keep
            # candidate must never appear.
            assert "arg160-current" not in titles
            assert "arg160-keep-no-emb" not in titles
            assert "arg160-nonkeep-near" not in titles

            # All three embedded Keep assets surface somewhere.
            assert "arg160-keep-near" in titles
            assert "arg160-keep-mid" in titles
            assert "arg160-keep-far" in titles

            # Ordering among seeded Keep assets matches proximity to anchor.
            idx_near = titles.index("arg160-keep-near")
            idx_mid = titles.index("arg160-keep-mid")
            idx_far = titles.index("arg160-keep-far")
            assert idx_near < idx_mid < idx_far

        async with Session() as session:
            # When the current item has no embedding, the comparison can't be
            # anchored — the subsection is empty (Codex P2: no global fallback).
            similar_no_anchor = await _fetch_similar(
                session, keep_no_emb.id, limit=5000
            )
            no_anchor_titles = [s.title for s in similar_no_anchor]
            assert "arg160-keep-near" not in no_anchor_titles
            assert "arg160-keep-mid" not in no_anchor_titles
            assert "arg160-keep-far" not in no_anchor_titles
    finally:
        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_item_detail_returns_digest() -> None:
    """ARG-182: fetch_item_detail exposes the digest column (ARG-173/Task 1)."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_id: uuid.UUID | None = None
    try:
        async with Session() as session:
            item = TechItem(
                title="arg182-digest",
                source_url=f"https://example.com/arg182/{uuid.uuid4()}",
                raw_content="x" * 2000,
                summary="s",
                digest="롱폼 본문",
                category=CategoryType.ALPHA,
                trust_score=0.7,
            )
            session.add(item)
            await session.flush()
            seeded_id = item.id
            await session.commit()

        async with Session() as session:
            view = await fetch_item_detail(session, seeded_id)
            assert view is not None
            assert view.digest == "롱폼 본문"
    finally:
        async with Session() as session:
            if seeded_id is not None:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id == seeded_id)
                )
            await session.commit()
        await engine.dispose()


def test_similar_limit_default_is_five() -> None:
    from argos.web.services.detail import SIMILAR_LIMIT

    assert SIMILAR_LIMIT == 5


def test_history_limit_default_is_ten() -> None:
    from argos.web.services.detail import HISTORY_LIMIT

    assert HISTORY_LIMIT == 10


@pytest.mark.asyncio
async def test_fetch_related_history_returns_empty_for_empty_tech_id_list() -> None:
    """No DB hit when the caller passes an empty list."""
    from argos.web.services.detail import _fetch_related_history

    class _Sentinel:
        async def execute(self, *args, **kwargs):
            raise AssertionError("DB must not be touched when tech_ids is empty")

    result = await _fetch_related_history(
        _Sentinel(), uuid.uuid4(), []
    )  # type: ignore[arg-type]
    assert result == []


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_related_history_returns_recent_rows_desc_for_seeded_tech() -> None:
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.detail import _fetch_related_history

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_tech_ids: list[uuid.UUID] = []
    try:
        from datetime import datetime, timezone

        async with Session() as session:
            tech = TechItem(
                title="arg161-history-anchor",
                source_url=f"https://example.com/arg161/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add(tech)
            await session.flush()
            seeded_tech_ids = [tech.id]
            asset = UserAsset(tech_id=tech.id, status=AssetStatus.KEEP)
            session.add(asset)
            await session.flush()
            session.add_all(
                [
                    # Real status transitions — these belong in 최근 변화.
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from="Tracking",
                        changed_to="Keep",
                        changed_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from="Keep",
                        changed_to="Archived",
                        changed_at=datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc),
                    ),
                    # Alert-dedup sentinels written by the Slack signal pipeline
                    # — must NOT surface in the timeline (would render raw noise).
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from="Keep",
                        changed_to="succession_alerted",
                        changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from=str(uuid.uuid4()),
                        changed_to="signal_matched",
                        changed_at=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
                    ),
                ]
            )
            await session.commit()
            tech_id = tech.id

        async with Session() as session:
            rows = await _fetch_related_history(session, tech_id, [tech_id])
            ours = [r for r in rows if r.tech_id == tech_id]
            # Only the two real status transitions survive; both sentinels
            # are filtered out.
            assert len(ours) == 2
            changed_tos = {r.changed_to for r in ours}
            assert changed_tos == {"Keep", "Archived"}
            assert "signal_matched" not in changed_tos
            assert "succession_alerted" not in changed_tos
            # Desc order: newest real transition first.
            assert ours[0].changed_to == "Archived"
            assert ours[1].changed_to == "Keep"
            assert ours[0].tech_title == "arg161-history-anchor"
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
async def test_fetch_signal_alerts_resolves_matched_item_and_skips_transitions() -> None:
    """signal_matched rows resolve to the matched TechItem (title + id),
    succession_alerted rows surface generically, and real status transitions
    are excluded — they belong to the 최근 변화 timeline."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.detail import _fetch_signal_alerts

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_tech_ids: list[uuid.UUID] = []
    try:
        from datetime import datetime, timezone

        async with Session() as session:
            anchor = TechItem(
                title="arg-signal-anchor",
                source_url=f"https://example.com/sig/{uuid.uuid4()}",
                raw_content="x",
            )
            matched = TechItem(
                title="arg-signal-matched-item",
                source_url=f"https://example.com/sig/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add_all([anchor, matched])
            await session.flush()
            seeded_tech_ids = [anchor.id, matched.id]
            asset = UserAsset(tech_id=anchor.id, status=AssetStatus.KEEP)
            session.add(asset)
            await session.flush()
            session.add_all(
                [
                    # signal_matched stores the matched item's id in changed_from.
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from=str(matched.id),
                        changed_to="signal_matched",
                        changed_at=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
                    ),
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from="Keep",
                        changed_to="succession_alerted",
                        changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                    # Real transition — must NOT appear among signal alerts.
                    TrackHistory(
                        user_asset_id=asset.id,
                        changed_from="Tracking",
                        changed_to="Keep",
                        changed_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                ]
            )
            await session.commit()
            anchor_id = anchor.id
            matched_id = matched.id

        async with Session() as session:
            alerts = await _fetch_signal_alerts(session, anchor_id, [anchor_id])
            # Only the two alert rows; the status transition is excluded.
            kinds = [a.kind for a in alerts]
            assert kinds == ["signal", "succession"]  # desc by changed_at

            sig = alerts[0]
            assert sig.kind == "signal"
            assert sig.matched_tech_id == matched_id
            assert sig.matched_title == "arg-signal-matched-item"

            succ = alerts[1]
            assert succ.kind == "succession"
            assert succ.matched_tech_id is None
            assert succ.matched_title is None
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
async def test_fetch_signal_alerts_prioritizes_current_item_before_limit() -> None:
    """The viewed item's own alert must win a tight limit even when a similar
    asset has a newer alert — otherwise tapping an active card shows no
    explanation for it."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset
    from argos.web.services.detail import _fetch_signal_alerts

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_tech_ids: list[uuid.UUID] = []
    try:
        from datetime import datetime, timezone

        async with Session() as session:
            current = TechItem(
                title="arg-prio-current",
                source_url=f"https://example.com/prio/{uuid.uuid4()}",
                raw_content="x",
            )
            similar = TechItem(
                title="arg-prio-similar",
                source_url=f"https://example.com/prio/{uuid.uuid4()}",
                raw_content="x",
            )
            cur_match = TechItem(
                title="arg-prio-current-match",
                source_url=f"https://example.com/prio/{uuid.uuid4()}",
                raw_content="x",
            )
            sim_match = TechItem(
                title="arg-prio-similar-match",
                source_url=f"https://example.com/prio/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add_all([current, similar, cur_match, sim_match])
            await session.flush()
            seeded_tech_ids = [current.id, similar.id, cur_match.id, sim_match.id]
            cur_asset = UserAsset(tech_id=current.id, status=AssetStatus.KEEP)
            sim_asset = UserAsset(tech_id=similar.id, status=AssetStatus.KEEP)
            session.add_all([cur_asset, sim_asset])
            await session.flush()
            session.add_all(
                [
                    # Current item's alert is OLDER…
                    TrackHistory(
                        user_asset_id=cur_asset.id,
                        changed_from=str(cur_match.id),
                        changed_to="signal_matched",
                        changed_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                    # …the similar asset's alert is NEWER.
                    TrackHistory(
                        user_asset_id=sim_asset.id,
                        changed_from=str(sim_match.id),
                        changed_to="signal_matched",
                        changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                ]
            )
            await session.commit()
            current_id = current.id
            similar_id = similar.id
            cur_match_id = cur_match.id

        async with Session() as session:
            # limit=1: pure recency would return the similar asset's newer alert,
            # but prioritization must surface the current item's own alert.
            alerts = await _fetch_signal_alerts(
                session, current_id, [current_id, similar_id], limit=1
            )
            assert len(alerts) == 1
            assert alerts[0].matched_tech_id == cur_match_id
            assert alerts[0].matched_title == "arg-prio-current-match"
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
async def test_fetch_item_detail_loads_predecessors_and_successors() -> None:
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.tech_succession import RelationType, TechSuccession

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_ids: list[uuid.UUID] = []
    try:
        async with Session() as session:
            anchor = TechItem(
                title="arg159-anchor",
                source_url=f"https://example.com/arg159/{uuid.uuid4()}",
                raw_content="x",
            )
            parent = TechItem(
                title="arg159-parent",
                source_url=f"https://example.com/arg159/{uuid.uuid4()}",
                raw_content="x",
            )
            child = TechItem(
                title="arg159-child",
                source_url=f"https://example.com/arg159/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add_all([anchor, parent, child])
            await session.flush()
            seeded_ids = [anchor.id, parent.id, child.id]

            session.add_all(
                [
                    TechSuccession(
                        predecessor_id=parent.id,
                        successor_id=anchor.id,
                        relation_type=RelationType.REPLACE,
                        reasoning="parent replaced",
                    ),
                    TechSuccession(
                        predecessor_id=anchor.id,
                        successor_id=child.id,
                        relation_type=RelationType.ENHANCE,
                        reasoning="child enhances",
                    ),
                ]
            )
            await session.commit()
            anchor_id = anchor.id

        async with Session() as session:
            view = await fetch_item_detail(session, anchor_id)
            assert view is not None
            titles_pred = [p.title for p in view.predecessors]
            titles_succ = [s.title for s in view.successors]
            assert "arg159-parent" in titles_pred
            assert "arg159-child" in titles_succ
            pred = next(p for p in view.predecessors if p.title == "arg159-parent")
            assert pred.relation_type == RelationType.REPLACE
            assert pred.reasoning == "parent replaced"
            succ = next(s for s in view.successors if s.title == "arg159-child")
            assert succ.relation_type == RelationType.ENHANCE
    finally:
        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_item_detail_fills_timeline_only_for_keep_assets() -> None:
    """ARG-208: a Keep asset's detail view gets the full unified timeline
    (fetch_timeline, limit=None); a non-asset item's timeline stays empty and
    keeps populating the older related_history/signal_alerts fields."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import TechItem
    from argos.models.track_history import TrackHistory
    from argos.models.user_asset import AssetStatus, UserAsset

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_ids: list[uuid.UUID] = []
    try:
        from datetime import datetime, timezone

        async with Session() as session:
            keep_item = TechItem(
                title="arg208-keep-item",
                source_url=f"https://example.com/arg208/{uuid.uuid4()}",
                raw_content="x",
            )
            non_asset_item = TechItem(
                title="arg208-non-asset-item",
                source_url=f"https://example.com/arg208/{uuid.uuid4()}",
                raw_content="x",
            )
            session.add_all([keep_item, non_asset_item])
            await session.flush()
            seeded_ids = [keep_item.id, non_asset_item.id]
            asset = UserAsset(tech_id=keep_item.id, status=AssetStatus.KEEP)
            session.add(asset)
            await session.flush()
            session.add(
                TrackHistory(
                    user_asset_id=asset.id,
                    changed_from="Tracking",
                    changed_to="Keep",
                    changed_at=datetime(2026, 6, 10, 9, 30, tzinfo=timezone.utc),
                )
            )
            await session.commit()
            keep_item_id = keep_item.id
            non_asset_item_id = non_asset_item.id

        async with Session() as session:
            view = await fetch_item_detail(session, keep_item_id)
            assert view is not None
            assert view.timeline  # non-empty for a Keep asset with events
            assert view.timeline[0].kind == "status"
            assert view.timeline[0].changed_to == "Keep"
            # Superseded by timeline for Keep assets — left empty rather
            # than duplicating data the unified fragment already renders.
            assert view.related_history == []
            assert view.signal_alerts == []

            view2 = await fetch_item_detail(session, non_asset_item_id)
            assert view2 is not None
            assert view2.timeline == []
    finally:
        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()
