"""Detail-service tests for the 신뢰도 구성 breakdown (ARG-211 Task 3).

``fetch_item_detail`` must expose ``trust_rubric`` (dict) and
``corroboration_count`` (int) on ``ItemDetailView`` so the item detail page
can render the 5-field evidence rubric under the trust dial. Legacy rows
(pre-ARG-206, ``trust_rubric IS NULL``) must surface ``trust_rubric is None``
so the template can render nothing (or a legacy caption) instead of a
half-populated breakdown.
"""
from __future__ import annotations

import uuid

import pytest

from argos.config import settings
from argos.web.services.detail import ItemDetailView, fetch_item_detail
from tests.conftest import db_reachable as _db_reachable


_DB_URL: str = settings.database_url


pytestmark_db = pytest.mark.skipif(
    not _db_reachable(_DB_URL),
    reason="pgvector DB not reachable — skipping ARG-211 DB-backed tests",
)


def test_item_detail_view_trust_fields_default_to_none() -> None:
    """Existing call sites that don't pass trust_rubric/corroboration_count
    must stay unaffected — both fields default to None."""
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
    assert view.trust_rubric is None
    assert view.corroboration_count is None


@pytestmark_db
@pytest.mark.asyncio
async def test_fetch_item_detail_returns_trust_rubric_and_corroboration_count() -> None:
    from sqlalchemy import delete as sa_delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.tech_item import CategoryType, TechItem

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    seeded_id: uuid.UUID | None = None
    rubric = {
        "is_primary_source": True,
        "has_evidence_links": True,
        "has_concrete_numbers": False,
        "claim_evidence_balance": "balanced",
        "marketing_intensity": "low",
    }
    try:
        async with Session() as session:
            item = TechItem(
                title="arg211-trust-breakdown",
                source_url=f"https://example.com/arg211/{uuid.uuid4()}",
                raw_content="raw",
                summary="s",
                category=CategoryType.ALPHA,
                trust_score=0.81,
                trust_rubric=rubric,
                corroboration_count=3,
            )
            session.add(item)
            await session.flush()
            seeded_id = item.id
            await session.commit()

        async with Session() as session:
            view = await fetch_item_detail(session, seeded_id)
            assert view is not None
            assert view.trust_rubric == rubric
            assert view.corroboration_count == 3
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
async def test_fetch_item_detail_legacy_row_has_none_trust_rubric() -> None:
    """A legacy row (pre-ARG-206, trust_rubric never backfilled) must surface
    trust_rubric=None rather than an empty dict or raising."""
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
                title="arg211-legacy",
                source_url=f"https://example.com/arg211/{uuid.uuid4()}",
                raw_content="raw",
                summary="s",
                category=CategoryType.ALPHA,
                trust_score=0.5,
                trust_rubric=None,
                corroboration_count=None,
            )
            session.add(item)
            await session.flush()
            seeded_id = item.id
            await session.commit()

        async with Session() as session:
            view = await fetch_item_detail(session, seeded_id)
            assert view is not None
            assert view.trust_rubric is None
            assert view.corroboration_count is None
    finally:
        async with Session() as session:
            if seeded_id is not None:
                await session.execute(
                    sa_delete(TechItem).where(TechItem.id == seeded_id)
                )
            await session.commit()
        await engine.dispose()
