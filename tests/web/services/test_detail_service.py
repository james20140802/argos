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
    finally:
        async with Session() as session:
            if seeded_id is not None:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id == seeded_id)
                )
            await session.commit()
        await engine.dispose()
