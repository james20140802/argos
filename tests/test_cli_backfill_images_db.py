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


@pytest.mark.asyncio
async def test_backfill_upgrade_favicons(monkeypatch):
    """``--upgrade-favicons`` replaces a favicon with a real og:image only.

    - A favicon row whose re-crawl yields a real image is upgraded.
    - A favicon row whose re-crawl yields only a favicon is left untouched.
    - A row already holding a real image is never selected.
    """
    upgrade_id = uuid.uuid4()      # favicon → real image (upgrade)
    stuck_id = uuid.uuid4()        # favicon → favicon (leave as-is)
    real_id = uuid.uuid4()         # already real (never selected)
    real_image = "https://cdn.example.com/og-card.png"
    kept_real = "https://cdn.example.com/already-real.jpg"

    engine, factory = await _session_factory()
    try:
        async with factory() as session:
            session.add(
                TechItem(
                    id=upgrade_id,
                    title="upgrade fixture",
                    source_url=f"https://up-arg179.test/{upgrade_id}",
                    raw_content="fixture",
                    category=CategoryType.ALPHA,
                    image_url="https://up-arg179.test/favicon.ico",
                )
            )
            session.add(
                TechItem(
                    id=stuck_id,
                    title="stuck fixture",
                    source_url=f"https://stuck-arg179.test/{stuck_id}",
                    raw_content="fixture",
                    category=CategoryType.ALPHA,
                    image_url="https://stuck-arg179.test/favicon.ico",
                )
            )
            session.add(
                TechItem(
                    id=real_id,
                    title="already-real fixture",
                    source_url=f"https://real-arg179.test/{real_id}",
                    raw_content="fixture",
                    category=CategoryType.MAINSTREAM,
                    image_url=kept_real,
                )
            )
            await session.commit()

        # Re-crawl yields a real image for the upgrade row, a favicon for the
        # stuck row (resolver found nothing better than the domain favicon).
        async def _fake_refetch(source_url: str):
            if str(upgrade_id) in source_url:
                return real_image
            return "https://stuck-arg179.test/favicon.ico"

        import argos.cli as cli

        # ``_backfill_images`` uses the global ``AsyncSessionLocal`` engine. Its
        # pool may hold a connection bound to a prior test's event loop; dispose
        # it so the backfill reconnects on this test's loop.
        from argos.database import engine as _global_engine

        await _global_engine.dispose()

        monkeypatch.setattr(cli, "_refetch_image_url", _fake_refetch)
        await cli._backfill_images(upgrade_favicons=True)

        async with factory() as session:
            rows = {
                r.id: r.image_url
                for r in (
                    await session.execute(
                        select(TechItem).where(
                            TechItem.id.in_([upgrade_id, stuck_id, real_id])
                        )
                    )
                ).scalars()
            }

        assert rows[upgrade_id] == real_image, "favicon row should upgrade to og:image"
        assert rows[stuck_id].endswith("/favicon.ico"), "favicon-only row stays favicon"
        assert rows[real_id] == kept_real, "real-image row is never selected/overwritten"

    finally:
        async with factory() as session:
            await session.execute(
                delete(TechItem).where(
                    TechItem.id.in_([upgrade_id, stuck_id, real_id])
                )
            )
            await session.commit()
        await engine.dispose()
