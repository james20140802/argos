"""Unit + integration tests for argos.web.services.detail (ARG-158)."""
from __future__ import annotations

import socket
import uuid

import pytest
from sqlalchemy.engine.url import make_url

from argos.config import settings
from argos.web.services.detail import ItemDetailView, fetch_item_detail


_DB_URL: str = settings.database_url


def _db_reachable(url: str) -> bool:
    parsed = make_url(url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


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
async def test_fetch_similar_ranks_seeded_items_by_proximity_to_keep_anchor() -> None:
    """Direct check on the similarity helper.

    Uses a generous LIMIT so the assertion ranks SEEDED items against
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
            keep_emb = [1.0] + [0.0] * 767
            very_similar = [0.99] + [0.01] * 767
            somewhat = [0.5] + [0.5] * 767
            # Far point on a perpendicular axis — cosine distance ≫ above.
            far = [0.0] * 767 + [1.0]

            keep_item = TechItem(
                title="arg160-keep-anchor",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=keep_emb,
            )
            anchor = TechItem(
                title="arg160-current",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=very_similar,
            )
            top = TechItem(
                title="arg160-top",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=very_similar,
            )
            mid = TechItem(
                title="arg160-mid",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=somewhat,
            )
            tail = TechItem(
                title="arg160-tail",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=far,
            )
            no_emb = TechItem(
                title="arg160-no-emb",
                source_url=f"https://example.com/arg160/{uuid.uuid4()}",
                raw_content="x",
                embedding=None,
            )
            session.add_all([keep_item, anchor, top, mid, tail, no_emb])
            await session.flush()
            seeded_ids = [
                keep_item.id,
                anchor.id,
                top.id,
                mid.id,
                tail.id,
                no_emb.id,
            ]
            session.add(UserAsset(tech_id=keep_item.id, status=AssetStatus.KEEP))
            await session.commit()
            anchor_id = anchor.id

        async with Session() as session:
            # Generous limit so seeded titles all surface regardless of
            # production data populating the top-K.
            similar = await _fetch_similar(session, anchor_id, limit=5000)
            titles = [s.title for s in similar]

            # Current item and the no-embedding row must never appear.
            assert "arg160-current" not in titles
            assert "arg160-no-emb" not in titles

            # All three embedded seeded candidates surface somewhere.
            assert "arg160-top" in titles
            assert "arg160-mid" in titles
            assert "arg160-tail" in titles

            # Ordering among seeded titles matches embedding proximity.
            idx_top = titles.index("arg160-top")
            idx_mid = titles.index("arg160-mid")
            idx_tail = titles.index("arg160-tail")
            assert idx_top < idx_mid < idx_tail
    finally:
        async with Session() as session:
            if seeded_ids:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id.in_(seeded_ids))
                )
            await session.commit()
        await engine.dispose()


def test_similar_limit_default_is_five() -> None:
    from argos.web.services.detail import SIMILAR_LIMIT

    assert SIMILAR_LIMIT == 5


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
