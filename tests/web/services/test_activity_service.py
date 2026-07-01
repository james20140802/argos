"""Tests for argos.web.services.activity — the 관측 신호 ticker.

The mock-level test runs without Postgres and pins the query shape (it must
filter to *current* Keep assets, mirroring the portfolio signal counts). The
DB-gated integration test proves the scoping end-to-end: a signal on an asset
that was later Archived must drop out of the ticker.

Skipped when the pgvector DB is unreachable (release.yml has no Postgres — see
CLAUDE.md "Release CI runs pytest with no DB").
"""
from __future__ import annotations

import socket
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import delete
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset
from argos.web.services.activity import fetch_activity

_DB_URL: str = settings.database_url


# ------------------------------------------------------------------ #
# Mock-level: query shape (no DB)
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_activity_query_filters_to_current_keep_assets() -> None:
    """The statement must scope to ``UserAsset.status == Keep`` so archived
    assets' stale signal rows never surface as live 관측 신호."""
    result = MagicMock()
    result.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    await fetch_activity(session)

    session.execute.assert_awaited_once()
    stmt = session.execute.await_args.args[0]
    sql = str(stmt).lower()
    # A predicate on the current asset status must be present.
    assert "user_assets.status" in sql


# ------------------------------------------------------------------ #
# DB integration: Keep-scoping end-to-end
# ------------------------------------------------------------------ #

def _db_reachable(url: str) -> bool:
    parsed = make_url(url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@asynccontextmanager
async def _session_ctx():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    try:
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_archived_asset_signal_drops_out_of_ticker():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping activity DB integration test")

    from argos.slack.services.track_check import SIGNAL_MATCHED

    kept_tech = uuid.uuid4()
    archived_tech = uuid.uuid4()
    kept_ua = uuid.uuid4()
    archived_ua = uuid.uuid4()
    now = datetime.now(timezone.utc)
    try:
        async with _session_ctx() as session:
            for tid, url in (
                (kept_tech, f"https://act.test/keep/{kept_tech}"),
                (archived_tech, f"https://act.test/arch/{archived_tech}"),
            ):
                session.add(
                    TechItem(
                        id=tid,
                        title=f"fixture {tid}",
                        source_url=url,
                        raw_content="fixture",
                        category=CategoryType.MAINSTREAM,
                        trust_score=0.5,
                    )
                )
            session.add(
                UserAsset(id=kept_ua, tech_id=kept_tech, status=AssetStatus.KEEP)
            )
            session.add(
                UserAsset(
                    id=archived_ua,
                    tech_id=archived_tech,
                    status=AssetStatus.ARCHIVED,
                )
            )
            # Each asset has one signal_matched row in its history.
            session.add(
                TrackHistory(
                    user_asset_id=kept_ua,
                    changed_from=str(uuid.uuid4()),
                    changed_to=SIGNAL_MATCHED,
                    changed_at=now,
                )
            )
            session.add(
                TrackHistory(
                    user_asset_id=archived_ua,
                    changed_from=str(uuid.uuid4()),
                    changed_to=SIGNAL_MATCHED,
                    changed_at=now,
                )
            )
            await session.commit()

        async with _session_ctx() as session:
            entries = await fetch_activity(session)

        tech_ids = {e.tech_id for e in entries}
        assert kept_tech in tech_ids
        assert archived_tech not in tech_ids
    finally:
        async with _session_ctx() as session:
            await session.execute(
                delete(TrackHistory).where(
                    TrackHistory.user_asset_id.in_([kept_ua, archived_ua])
                )
            )
            await session.execute(
                delete(UserAsset).where(
                    UserAsset.id.in_([kept_ua, archived_ua])
                )
            )
            await session.execute(
                delete(TechItem).where(
                    TechItem.id.in_([kept_tech, archived_tech])
                )
            )
            await session.commit()
