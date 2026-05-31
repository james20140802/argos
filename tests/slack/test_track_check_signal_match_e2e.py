"""ARG-119: End-to-end signal-match integration test.

Uses the real pgvector DB (Docker container must be running) with a real
AsyncSession backed by a NullPool engine (one fresh connection per test,
no pool sharing across event loops).

Slack client is mocked — the test verifies:
  1. match_signals returns exactly the high-similarity item (>0.85 threshold)
  2. post_signal_update calls chat_postMessage exactly once with expected text
  3. TrackHistory sentinel row (changed_to='signal_matched') is written to DB
  4. user_assets.last_monitored_at is updated
  5. Dedup: second match_signals call after dispatch returns [] (no-op)

Embeddings:
  - keep_emb      : random unit vector for the Keep-ed asset
  - similar_emb   : keep_emb + small noise; cosine sim > 0.99 → above 0.85
  - dissimilar_emb: orthogonal direction; cosine sim ≈ 0.05 → below 0.85

NullPool is used so each test gets a fresh DB connection on its own event loop,
avoiding the "another operation is in progress" asyncpg error that occurs when
pooled connections are shared across pytest-asyncio function-scope event loops.
"""
from __future__ import annotations

import socket
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem
from argos.models.track_history import TrackHistory
from argos.models.user_asset import AssetStatus, UserAsset
from argos.slack.services.track_check import (
    SIGNAL_MATCHED,
    match_signals,
    post_signal_update,
)

# Capture the database URL at import time so that wizard tests which call
# database.rebuild() (and mutate os.environ + settings.secrets) cannot
# change the URL seen by these E2E tests.
_DB_URL: str = settings.database_url


def _db_reachable(url: str) -> bool:
    """Return True if a TCP connection to the DB host:port succeeds quickly.

    Used to skip (rather than fail) these tests when no pgvector DB is
    running — e.g. the Release CI runner, which has no Docker service.
    """
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
    """Skip the E2E signal-match suite when the pgvector DB is unreachable.

    Synchronous on purpose: an async module-scoped fixture would reintroduce
    the cross-event-loop asyncpg trap that NullPool + per-test engines exist
    to avoid (see module docstring).
    """
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-119 E2E signal-match tests "
            "(start the Docker DB to run them)"
        )


# ---------------------------------------------------------------------------
# Per-test session factory (NullPool prevents cross-loop connection reuse)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_emb(v: np.ndarray) -> list[float]:
    return _unit(v).tolist()


def _emb_str(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.10f}" for x in v) + "]"


def _cosine_sim(a: list[float], b: list[float]) -> float:
    return float(np.dot(np.array(a), np.array(b)))  # unit vecs → dot = cos


# ---------------------------------------------------------------------------
# Embedding fixtures (module-scope — computed once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embeddings():
    rng = np.random.default_rng(42)
    keep_raw = rng.standard_normal(768)
    similar_raw = keep_raw + rng.standard_normal(768) * 0.05
    ortho_raw = rng.standard_normal(768)
    keep_unit = _unit(keep_raw)
    ortho_raw = ortho_raw - np.dot(ortho_raw, keep_unit) * keep_unit
    dissimilar_raw = _unit(ortho_raw)

    keep_emb = _make_emb(keep_raw)
    similar_emb = _make_emb(similar_raw)
    dissimilar_emb = _make_emb(dissimilar_raw)

    sim_s = _cosine_sim(keep_emb, similar_emb)
    sim_d = _cosine_sim(keep_emb, dissimilar_emb)
    assert sim_s > 0.85, f"similar_emb cosine sim too low: {sim_s:.4f}"
    assert sim_d <= 0.85, f"dissimilar_emb cosine sim too high: {sim_d:.4f}"

    return keep_emb, similar_emb, dissimilar_emb


# ---------------------------------------------------------------------------
# DB seeding helper
# ---------------------------------------------------------------------------


async def _seed(session, keep_emb, similar_emb, dissimilar_emb):
    """Insert Keep-ed asset + 2 new TechItems into the DB. Returns dict of IDs."""

    async def _add_item(title, url_tag, emb, category, trust):
        item = TechItem(
            title=title,
            source_url=f"https://e2e.internal/{url_tag}/{uuid.uuid4()}",
            raw_content=f"content for {title}",
            category=category,
            trust_score=trust,
        )
        session.add(item)
        await session.flush()
        await session.execute(
            text("UPDATE tech_items SET embedding = CAST(:emb AS vector) WHERE id = :id"),
            {"emb": _emb_str(emb), "id": str(item.id)},
        )
        return item

    keep_item = await _add_item("Keep Tech A", "keep", keep_emb, CategoryType.MAINSTREAM, 0.9)
    asset = UserAsset(tech_id=keep_item.id, status=AssetStatus.KEEP)
    session.add(asset)
    await session.flush()

    similar_item = await _add_item(
        "Similar Signal", "similar", similar_emb, CategoryType.ALPHA, 0.85
    )
    dissimilar_item = await _add_item(
        "Dissimilar Item", "dissimilar", dissimilar_emb, CategoryType.ALPHA, 0.5
    )

    return {
        "asset_id": asset.id,
        "keep_item_id": keep_item.id,
        "similar_item_id": similar_item.id,
        "dissimilar_item_id": dissimilar_item.id,
    }


def _mock_app():
    app = MagicMock()
    app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "1"})
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_signals_returns_only_similar_item(embeddings):
    """match_signals must return exactly 1 result above the 0.85 threshold."""
    keep_emb, similar_emb, dissimilar_emb = embeddings
    async with _session_ctx() as session:
        try:
            ids = await _seed(session, keep_emb, similar_emb, dissimilar_emb)
            new_ids = [ids["similar_item_id"], ids["dissimilar_item_id"]]

            matches = await match_signals(session, new_ids)

            assert len(matches) == 1, (
                f"Expected 1 match, got {len(matches)}: "
                f"{[m.new_item_title for m in matches]}"
            )
            m = matches[0]
            assert m.new_item_id == ids["similar_item_id"]
            assert m.user_asset_id == ids["asset_id"]
            assert m.keep_item_id == ids["keep_item_id"]
            assert m.keep_item_title == "Keep Tech A"
            assert m.new_item_title == "Similar Signal"
            assert m.similarity_score > 0.85
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_match_signals_none_scans_all_items(embeddings):
    """match_signals(None) scans all tech_items; our seeded pair is found once."""
    keep_emb, similar_emb, dissimilar_emb = embeddings
    async with _session_ctx() as session:
        try:
            ids = await _seed(session, keep_emb, similar_emb, dissimilar_emb)

            matches = await match_signals(session, None)

            our_matches = [m for m in matches if m.user_asset_id == ids["asset_id"]]
            assert len(our_matches) == 1
            assert our_matches[0].new_item_id == ids["similar_item_id"]
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_post_signal_update_sends_slack_and_writes_history(embeddings):
    """post_signal_update sends 1 Slack message and writes a TrackHistory sentinel row."""
    keep_emb, similar_emb, dissimilar_emb = embeddings
    async with _session_ctx() as session:
        try:
            ids = await _seed(session, keep_emb, similar_emb, dissimilar_emb)

            matches = await match_signals(
                session, [ids["similar_item_id"], ids["dissimilar_item_id"]]
            )
            assert len(matches) == 1

            app = _mock_app()
            await post_signal_update(app, "C_TEST", matches, session)
            await session.flush()

            # Verify Slack message
            app.client.chat_postMessage.assert_awaited_once()
            kwargs = app.client.chat_postMessage.await_args.kwargs
            assert kwargs["channel"] == "C_TEST"
            assert "🔭" in kwargs["text"]
            assert "Keep Tech A" in kwargs["text"]
            assert "Similar Signal" in kwargs["text"]

            # Verify TrackHistory row in DB
            result = await session.execute(
                select(TrackHistory).where(
                    TrackHistory.user_asset_id == ids["asset_id"],
                    TrackHistory.changed_to == SIGNAL_MATCHED,
                    TrackHistory.changed_from == str(ids["similar_item_id"]),
                )
            )
            rows = result.scalars().all()
            assert len(rows) == 1, f"Expected 1 TrackHistory row, got {len(rows)}"
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_post_signal_update_updates_last_monitored_at(embeddings):
    """user_assets.last_monitored_at must be set after a successful dispatch."""
    keep_emb, similar_emb, dissimilar_emb = embeddings
    async with _session_ctx() as session:
        try:
            ids = await _seed(session, keep_emb, similar_emb, dissimilar_emb)

            matches = await match_signals(session, [ids["similar_item_id"]])
            assert len(matches) == 1

            app = _mock_app()
            await post_signal_update(app, "C_TEST", matches, session)
            await session.flush()

            result = await session.execute(
                select(UserAsset).where(UserAsset.id == ids["asset_id"])
            )
            asset = result.scalar_one()
            assert asset.last_monitored_at is not None, "last_monitored_at was not updated"
            assert asset.last_monitored_at.tzinfo is not None  # timezone-aware
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_dedup_prevents_duplicate_slack_send(embeddings):
    """Second call to match_signals after dispatch must return [] (dedup)."""
    keep_emb, similar_emb, dissimilar_emb = embeddings
    async with _session_ctx() as session:
        try:
            ids = await _seed(session, keep_emb, similar_emb, dissimilar_emb)
            new_ids = [ids["similar_item_id"]]

            # First dispatch
            matches_first = await match_signals(session, new_ids)
            assert len(matches_first) == 1

            app = _mock_app()
            await post_signal_update(app, "C_TEST", matches_first, session)
            await session.flush()

            # Second call: sentinel is now in track_history → dedup
            matches_second = await match_signals(session, new_ids)
            assert matches_second == [], (
                f"Expected dedup to return [], got {len(matches_second)} matches"
            )

            await post_signal_update(app, "C_TEST", matches_second, session)
            assert app.client.chat_postMessage.await_count == 1, (
                f"Expected 1 total Slack send, got {app.client.chat_postMessage.await_count}"
            )
        finally:
            await session.rollback()
