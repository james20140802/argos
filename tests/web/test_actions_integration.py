"""ARG-142: Keep/Pass/Untrack DB integration smoke tests.

Mock-level coverage in ``test_actions_route.py`` already exercises the
route surface. This file guards what the mocks cannot see:

* The real ``user_assets`` upsert (unique constraint on ``tech_id``,
  ``ON CONFLICT DO NOTHING`` + ``SELECT ... FOR UPDATE``).
* ``track_history`` insertion paired with the user_asset transition in
  the same committed transaction.
* The ``AssetStatus`` enum ↔ Postgres ``asset_status`` enum mapping.

Skipped when the pgvector DB is unreachable (release.yml has no Postgres
service — see CLAUDE.md "Release CI runs pytest with no DB").

Each test inserts its own TechItem with a unique source_url and cleans
up all derived rows (track_history, user_assets, tech_items) afterward
so nothing leaks between tests — consistent with the project-wide
"DB reset pending" workaround.
"""
from __future__ import annotations

import socket
import uuid
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.testclient import TestClient

from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset
from argos.web.app import _get_session, build_web_app

# Captured at import so wizard tests that mutate settings can't change it.
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
            "pgvector DB not reachable — skipping ARG-142 action integration "
            "smoke tests (start the Docker DB to run them)"
        )


@asynccontextmanager
async def _session_ctx():
    """Yield a fresh session backed by a NullPool engine; dispose on exit.

    NullPool avoids the "another operation is in progress" asyncpg error
    that pooled connections trigger when shared across pytest-asyncio
    function-scope event loops.
    """
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    try:
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()


async def _insert_tech_item() -> uuid.UUID:
    tech_id = uuid.uuid4()
    async with _session_ctx() as session:
        session.add(
            TechItem(
                id=tech_id,
                title="ARG-142 integration fixture",
                source_url=f"https://example.test/arg-142/{tech_id}",
                raw_content="fixture",
                category=CategoryType.MAINSTREAM,
                trust_score=0.5,
            )
        )
        await session.commit()
    return tech_id


async def _cleanup(tech_id: uuid.UUID) -> None:
    """Delete history → user_asset → tech_item for this fixture."""
    async with _session_ctx() as session:
        user_asset_ids = (
            await session.execute(
                select(UserAsset.id).where(UserAsset.tech_id == tech_id)
            )
        ).scalars().all()
        if user_asset_ids:
            await session.execute(
                delete(TrackHistory).where(
                    TrackHistory.user_asset_id.in_(user_asset_ids)
                )
            )
            await session.execute(
                delete(UserAsset).where(UserAsset.id.in_(user_asset_ids))
            )
        await session.execute(delete(TechItem).where(TechItem.id == tech_id))
        await session.commit()


async def _fetch_asset(tech_id: uuid.UUID) -> UserAsset | None:
    async with _session_ctx() as session:
        return (
            await session.execute(
                select(UserAsset).where(UserAsset.tech_id == tech_id)
            )
        ).scalar_one_or_none()


async def _fetch_history(user_asset_id: uuid.UUID) -> list[TrackHistory]:
    async with _session_ctx() as session:
        return (
            await session.execute(
                select(TrackHistory)
                .where(TrackHistory.user_asset_id == user_asset_id)
                .order_by(TrackHistory.changed_at)
            )
        ).scalars().all()


def _client_with_real_db() -> TestClient:
    """Build a TestClient that calls the action routes with a real DB session.

    The action handlers commit inside the request; the override below
    yields a fresh NullPool-backed session per request to avoid sharing
    asyncpg connections across the test's event loops.
    """
    app = build_web_app()

    async def _override_session():
        engine = create_async_engine(_DB_URL, poolclass=NullPool)
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        try:
            async with factory() as session:
                yield session
        finally:
            await engine.dispose()

    app.dependency_overrides[_get_session] = _override_session
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.asyncio
async def test_keep_then_pass_upserts_user_asset_and_writes_history():
    """Keep creates the asset (no history); Pass on the same tech_id
    upserts the existing row and writes one Keep→Archived history row."""
    tech_id = await _insert_tech_item()
    try:
        client = _client_with_real_db()

        keep_resp = client.post(f"/items/{tech_id}/keep")
        assert keep_resp.status_code == 200, keep_resp.text

        asset_after_keep = await _fetch_asset(tech_id)
        assert asset_after_keep is not None
        assert asset_after_keep.status == AssetStatus.KEEP
        assert (await _fetch_history(asset_after_keep.id)) == []

        pass_resp = client.post(f"/items/{tech_id}/pass")
        assert pass_resp.status_code == 200, pass_resp.text

        asset_after_pass = await _fetch_asset(tech_id)
        assert asset_after_pass is not None
        # Upsert — same user_asset row, status flipped.
        assert asset_after_pass.id == asset_after_keep.id
        assert asset_after_pass.status == AssetStatus.ARCHIVED

        history = await _fetch_history(asset_after_pass.id)
        assert len(history) == 1
        assert history[0].changed_from == AssetStatus.KEEP.value
        assert history[0].changed_to == AssetStatus.ARCHIVED.value
    finally:
        await _cleanup(tech_id)


@pytest.mark.asyncio
async def test_pass_first_then_repeat_pass_toggles_off():
    """First Pass creates an Archived row (no history); pressing Pass again
    toggles it OFF — the user_asset is deleted and the item is untriaged."""
    tech_id = await _insert_tech_item()
    try:
        client = _client_with_real_db()

        first = client.post(f"/items/{tech_id}/pass")
        assert first.status_code == 200, first.text

        asset = await _fetch_asset(tech_id)
        assert asset is not None
        assert asset.status == AssetStatus.ARCHIVED
        assert (await _fetch_history(asset.id)) == []

        # Repeat — toggle-off. The route returns 200 (the re-rendered untriaged
        # card) and the user_asset is removed entirely.
        repeat = client.post(f"/items/{tech_id}/pass")
        assert repeat.status_code == 200, repeat.text

        async with _session_ctx() as session:
            remaining = (
                await session.execute(
                    select(UserAsset).where(UserAsset.tech_id == tech_id)
                )
            ).scalars().all()
        assert remaining == []
    finally:
        await _cleanup(tech_id)


@pytest.mark.asyncio
async def test_toggle_off_cascade_deletes_history_rows():
    """Regression: toggling off an asset that already has track_history rows
    must succeed (the FK cascades), not 500.

    Keep → Pass writes one Keep→Archived history row; pressing Pass again then
    toggles the (Archived) asset off. Deleting the user_asset must cascade the
    history row instead of the ORM nulling its NOT NULL user_asset_id — the bug
    that surfaced only for previously-transitioned items."""
    tech_id = await _insert_tech_item()
    try:
        client = _client_with_real_db()

        assert client.post(f"/items/{tech_id}/keep").status_code == 200
        pass_resp = client.post(f"/items/{tech_id}/pass")
        assert pass_resp.status_code == 200, pass_resp.text

        asset = await _fetch_asset(tech_id)
        assert asset is not None and asset.status == AssetStatus.ARCHIVED
        # A transition happened, so there is a history row to cascade.
        assert len(await _fetch_history(asset.id)) == 1
        asset_id = asset.id

        # Toggle off — the asset and its history row are both removed.
        repeat = client.post(f"/items/{tech_id}/pass")
        assert repeat.status_code == 200, repeat.text

        assert await _fetch_asset(tech_id) is None
        assert await _fetch_history(asset_id) == []
    finally:
        await _cleanup(tech_id)


@pytest.mark.asyncio
async def test_untrack_archives_kept_asset_and_writes_history():
    """Untrack on a previously Kept asset archives it via the user_asset
    id route and writes one Keep→Archived history row."""
    tech_id = await _insert_tech_item()
    try:
        client = _client_with_real_db()

        keep_resp = client.post(f"/items/{tech_id}/keep")
        assert keep_resp.status_code == 200, keep_resp.text

        asset = await _fetch_asset(tech_id)
        assert asset is not None
        assert asset.status == AssetStatus.KEEP

        untrack_resp = client.post(f"/assets/{asset.id}/untrack")
        assert untrack_resp.status_code == 200, untrack_resp.text
        # Empty body — HTMX outerHTML swap removes the card.
        assert untrack_resp.text == ""

        after = await _fetch_asset(tech_id)
        assert after is not None
        assert after.id == asset.id
        assert after.status == AssetStatus.ARCHIVED

        history = await _fetch_history(asset.id)
        assert len(history) == 1
        assert history[0].changed_from == AssetStatus.KEEP.value
        assert history[0].changed_to == AssetStatus.ARCHIVED.value
    finally:
        await _cleanup(tech_id)
