"""ARG-179 T4: DB integration test for `argos backfill-images`.

Verifies the critical non-overwrite guarantee and the NULL-only SELECT:
- A TechItem with image_url IS NULL gets filled with the domain favicon.
- A TechItem with a non-NULL image_url is NOT touched.

Skipped when the pgvector DB is unreachable (release.yml has no Postgres
service — see CLAUDE.md "Release CI runs pytest with no DB").
"""
from __future__ import annotations

import socket
import uuid

import pytest
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem

# Captured at import time so wizard tests that mutate settings can't change it.
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


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-179 backfill-images DB "
            "integration test (start the Docker DB to run it)"
        )


async def _session_factory():
    """Return a fresh NullPool-backed sessionmaker."""
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    return engine, factory


@pytest.mark.asyncio
async def test_backfill_images_null_only_and_non_overwrite():
    """Default backfill fills NULL rows; non-NULL rows are NEVER overwritten."""
    # ---------- fixture setup ----------
    null_id = uuid.uuid4()
    nonnull_id = uuid.uuid4()
    existing_image = "https://example.com/kept-image.png"

    engine, factory = await _session_factory()
    try:
        async with factory() as session:
            # Row 1: image_url IS NULL — should be filled
            session.add(
                TechItem(
                    id=null_id,
                    title="ARG-179 null image fixture",
                    source_url=f"https://example-arg179.test/null/{null_id}",
                    raw_content="fixture",
                    category=CategoryType.MAINSTREAM,
                    trust_score=0.5,
                    image_url=None,
                )
            )
            # Row 2: image_url already set — must NOT be overwritten
            session.add(
                TechItem(
                    id=nonnull_id,
                    title="ARG-179 nonnull image fixture",
                    source_url=f"https://example-arg179.test/nonnull/{nonnull_id}",
                    raw_content="fixture",
                    category=CategoryType.MAINSTREAM,
                    trust_score=0.5,
                    image_url=existing_image,
                )
            )
            await session.commit()

        # ---------- run the CLI backfill ----------
        from argos.cli import _backfill_images  # import after insert so session is fresh
        await _backfill_images(refetch=False)

        # ---------- assertions ----------
        async with factory() as session:
            null_row = (
                await session.execute(select(TechItem).where(TechItem.id == null_id))
            ).scalar_one()
            nonnull_row = (
                await session.execute(select(TechItem).where(TechItem.id == nonnull_id))
            ).scalar_one()

        # The NULL row should now carry the favicon URL for its domain
        assert null_row.image_url is not None
        assert null_row.image_url == "https://example-arg179.test/favicon.ico", (
            f"Expected favicon URL, got: {null_row.image_url}"
        )

        # The non-NULL row must be completely unchanged
        assert nonnull_row.image_url == existing_image, (
            f"Non-NULL image_url was overwritten! Got: {nonnull_row.image_url}"
        )

    finally:
        # ---------- cleanup ----------
        async with factory() as session:
            await session.execute(
                delete(TechItem).where(TechItem.id.in_([null_id, nonnull_id]))
            )
            await session.commit()
        await engine.dispose()
