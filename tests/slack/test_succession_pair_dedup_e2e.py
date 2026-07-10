"""ARG-204: End-to-end pair-dedup integration tests for succession alerts.

Uses the real Postgres DB (Docker container must be running) with a real
AsyncSession backed by a NullPool engine (one fresh connection per test, no
pool sharing across event loops) — mirrors
tests/slack/test_track_check_signal_match_e2e.py (ARG-119), which proved out
the same pattern for match_signals.

Covers the behavior that cannot be honestly demonstrated with mocked
sessions (mocks only return whatever the test scripts them to return, so
they can't prove the SQL predicate actually filters correctly against real
committed data):

  1. A Keep-ed asset with two successions (P->S1, P->S2) yields two alerts,
     each carrying its own successor_id — dedup is per pair, not per asset.
  2. After post_track_update for the S1 alert, a re-check only returns S2 —
     the (asset, S1) pair is suppressed but (asset, S2) is not.
  3. A legacy 'succession_alerted' row written with changed_from='Keep' (the
     pre-ARG-204 asset-level encoding) does not suppress any pair — the
     asset re-alerts once under the new pair-based predicate. This is the
     documented, intended migration behavior (AC 2), not a special case in
     code.

Note on commit vs. flush: tests below use ``session.flush()`` (not
``session.commit()``) after writing the dedup TrackHistory row, then
``session.rollback()`` in a ``finally`` block. Because check_succession and
post_track_update run on the *same* session/connection, a flush is
sufficient for the NOT EXISTS predicate to see the write (Postgres
read-your-own-writes within an open transaction) — and rollback keeps the
DB clean for repeated local test runs, matching the established convention
in test_track_check_signal_match_e2e.py.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import RelationType, TechSuccession
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset
from argos.slack.services.track_check import (
    SUCCESSION_ALERTED,
    check_succession,
    post_track_update,
)
from tests.conftest import db_reachable as _db_reachable

# Capture the database URL at import time — see test_track_check_signal_match_e2e.py
# for why (wizard tests mutate settings.secrets via database.rebuild()).
_DB_URL: str = settings.database_url


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    """Skip this DB-backed suite when the pgvector DB is unreachable."""
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-204 succession pair-dedup "
            "E2E tests (start the Docker DB to run them)"
        )


@asynccontextmanager
async def _session_ctx():
    """Yield a fresh session backed by a NullPool engine; dispose on exit."""
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    try:
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()


async def _seed_pair(session) -> dict:
    """Seed one Keep-ed predecessor with two successions (P->S1, P->S2)."""

    async def _mk_item(title: str, tag: str) -> TechItem:
        item = TechItem(
            title=title,
            source_url=f"https://e2e.internal/{tag}/{uuid.uuid4()}",
            raw_content=f"content for {title}",
            category=CategoryType.MAINSTREAM,
            trust_score=0.9,
        )
        session.add(item)
        await session.flush()
        return item

    predecessor = await _mk_item("Predecessor Tech", "pred")
    successor_1 = await _mk_item("Successor One", "succ1")
    successor_2 = await _mk_item("Successor Two", "succ2")

    asset = UserAsset(tech_id=predecessor.id, status=AssetStatus.KEEP)
    session.add(asset)
    await session.flush()

    succession_1 = TechSuccession(
        predecessor_id=predecessor.id,
        successor_id=successor_1.id,
        relation_type=RelationType.REPLACE,
        reasoning="e2e seed",
    )
    succession_2 = TechSuccession(
        predecessor_id=predecessor.id,
        successor_id=successor_2.id,
        relation_type=RelationType.ENHANCE,
        reasoning="e2e seed",
    )
    session.add_all([succession_1, succession_2])
    await session.flush()

    return {
        "asset_id": asset.id,
        "predecessor_id": predecessor.id,
        "successor_1_id": successor_1.id,
        "successor_2_id": successor_2.id,
    }


def _mock_app():
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_succession_returns_one_alert_per_successor():
    """A Keep-ed asset with two distinct successors yields two alerts, each
    carrying its own successor_id (AC 1: dedup is per pair, not per asset)."""
    async with _session_ctx() as session:
        try:
            ids = await _seed_pair(session)
            new_ids = [ids["successor_1_id"], ids["successor_2_id"]]

            alerts = await check_succession(session, new_ids)

            successor_ids = {a.successor_id for a in alerts}
            assert successor_ids == {ids["successor_1_id"], ids["successor_2_id"]}
            assert all(a.user_asset_id == ids["asset_id"] for a in alerts)
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_pair_dedup_suppresses_only_alerted_successor():
    """After post_track_update fires for the S1 alert, re-checking must
    suppress (asset, S1) while still returning (asset, S2) — proving the
    NOT EXISTS predicate correlates on successor_id, not just user_asset_id."""
    async with _session_ctx() as session:
        try:
            ids = await _seed_pair(session)
            new_ids = [ids["successor_1_id"], ids["successor_2_id"]]

            alerts = await check_succession(session, new_ids)
            assert len(alerts) == 2

            alert_s1 = next(
                a for a in alerts if a.successor_id == ids["successor_1_id"]
            )

            app = _mock_app()
            await post_track_update(app, "C_TEST", [alert_s1], session)
            await session.flush()

            alerts_2 = await check_succession(session, new_ids)
            successor_ids_2 = {a.successor_id for a in alerts_2}
            assert successor_ids_2 == {ids["successor_2_id"]}, (
                f"Expected only S2 to remain, got {successor_ids_2}"
            )

            # And the TrackHistory row itself carries the pair encoding
            # (changed_from = str(successor_id), not the legacy 'Keep').
            result = await session.execute(
                select(TrackHistory).where(
                    TrackHistory.user_asset_id == ids["asset_id"],
                    TrackHistory.changed_to == SUCCESSION_ALERTED,
                )
            )
            history_rows = result.scalars().all()
            assert len(history_rows) == 1
            assert history_rows[0].changed_from == str(ids["successor_1_id"])
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_legacy_keep_encoded_row_does_not_suppress_new_pair_dedup():
    """Pre-ARG-204 rows recorded changed_from='Keep' (asset-level encoding).
    Under the new pair-based predicate (changed_from == str(successor_id)),
    such legacy rows match no successor UUID, so the asset naturally
    re-alerts once for every successor. This is intended (AC 2) — asserted
    here rather than special-cased in code."""
    async with _session_ctx() as session:
        try:
            ids = await _seed_pair(session)
            new_ids = [ids["successor_1_id"], ids["successor_2_id"]]

            # Simulate a legacy pre-ARG-204 TrackHistory row for this asset.
            session.add(
                TrackHistory(
                    user_asset_id=ids["asset_id"],
                    changed_from=AssetStatus.KEEP.value,
                    changed_to=SUCCESSION_ALERTED,
                )
            )
            await session.flush()

            alerts = await check_succession(session, new_ids)
            successor_ids = {a.successor_id for a in alerts}
            assert successor_ids == {
                ids["successor_1_id"],
                ids["successor_2_id"],
            }, "Legacy 'Keep'-encoded row must not suppress pair-based alerts"
        finally:
            await session.rollback()
