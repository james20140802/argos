"""ARG-212: feed_score batch ranking tests.

Pure scoring-function tests are DB-free. The batch `recompute_feed_scores`
test uses the real pgvector DB, following the same NullPool/module-skip
pattern as ``tests/brain/test_corroboration.py``.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus, UserAsset
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url


def test_recency_decay_half_life():
    from argos.brain.feed_ranking import recency_decay

    assert recency_decay(0.0, 48.0) == 1.0
    assert abs(recency_decay(48.0, 48.0) - 0.5) < 1e-9


def test_trending_score_monotonic_bounded():
    from argos.brain.feed_ranking import trending_score

    assert trending_score(0) == 0.0
    assert 0.0 < trending_score(1) <= 1.0
    assert trending_score(5) >= trending_score(1)


def test_profile_vector_none_when_no_keep():
    from argos.brain.feed_ranking import compute_profile_vector

    now = datetime.now(timezone.utc)
    assert (
        compute_profile_vector([], [], now=now, half_life_hours=48.0, pass_weight=0.3)
        is None
    )


def test_profile_vector_keep_minus_pass():
    from argos.brain.feed_ranking import compute_profile_vector

    now = datetime.now(timezone.utc)
    keep = [([1.0, 0.0], now)]  # 단일 Keep, 최신 → weight 1
    passv = [[0.0, 1.0]]
    vec = compute_profile_vector(keep, passv, now=now, half_life_hours=48.0, pass_weight=0.3)
    assert vec is not None
    assert vec[0] > 0 and vec[1] < 0  # keep 방향 +, pass 방향 −


# ---------------------------------------------------------------------------
# recompute_feed_scores — DB-backed batch tests (self-skip if DB unreachable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-212 feed_ranking batch "
            "tests (start the Docker DB to run them)"
        )


@asynccontextmanager
async def _session_ctx():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    try:
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_emb(v: np.ndarray) -> list[float]:
    return _unit(v).tolist()


def _emb_str(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.10f}" for x in v) + "]"


@pytest.fixture(scope="module")
def embeddings():
    rng = np.random.default_rng(11)
    base_raw = rng.standard_normal(768)
    similar_raw = base_raw + rng.standard_normal(768) * 0.02
    ortho_raw = rng.standard_normal(768)
    base_unit = _unit(base_raw)
    ortho_raw = ortho_raw - np.dot(ortho_raw, base_unit) * base_unit

    return _make_emb(base_raw), _make_emb(similar_raw), _make_emb(ortho_raw)


def _uniq(tag: str) -> str:
    return f"https://{tag}/{uuid.uuid4()}"


async def _add_item(
    session,
    title,
    url,
    emb,
    *,
    trust_score=0.5,
    corroboration_count=0,
    published_at=None,
):
    item = TechItem(
        title=title,
        source_url=url,
        raw_content=f"content for {title}",
        category=CategoryType.ALPHA,
        trust_score=trust_score,
        corroboration_count=corroboration_count,
        published_at=published_at or datetime.now(timezone.utc),
    )
    session.add(item)
    await session.flush()
    await session.execute(
        text("UPDATE tech_items SET embedding = CAST(:emb AS vector) WHERE id = :id"),
        {"emb": _emb_str(emb), "id": str(item.id)},
    )
    return item


async def _keep(session, tech_item) -> None:
    session.add(UserAsset(tech_id=tech_item.id, status=AssetStatus.KEEP))
    await session.flush()


async def _get_item(session, item_id):
    result = await session.execute(
        select(TechItem)
        .where(TechItem.id == item_id)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one()


@pytest.mark.asyncio
async def test_recompute_feed_scores_rewards_profile_similarity(embeddings):
    """A new item (B) similar to a Keep-ed item's embedding must score higher
    than an unrelated new item (C) — same trust/corroboration/recency for
    both, so the only differentiator is the profile-similarity term."""
    base_emb, similar_emb, ortho_emb = embeddings
    from argos.brain.feed_ranking import recompute_feed_scores

    async with _session_ctx() as session:
        try:
            kept = await _add_item(session, "Kept-A", _uniq("feed-a.com"), base_emb)
            await _keep(session, kept)

            b = await _add_item(
                session, "Similar-B", _uniq("feed-b.com"), similar_emb,
                trust_score=0.4, corroboration_count=2,
            )
            c = await _add_item(
                session, "Unrelated-C", _uniq("feed-c.com"), ortho_emb,
                trust_score=0.4, corroboration_count=2,
            )

            updated = await recompute_feed_scores(session)
            assert updated >= 3

            b_after = await _get_item(session, b.id)
            c_after = await _get_item(session, c.id)
            assert b_after.feed_score is not None
            assert c_after.feed_score is not None
            assert b_after.feed_score > c_after.feed_score
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_recompute_feed_scores_no_keep_no_exception(embeddings):
    """With zero Keep-ed assets (cold start), recompute_feed_scores must
    complete without raising and still score every item (profile term 0.0)."""
    _base_emb, _similar_emb, ortho_emb = embeddings
    from argos.brain.feed_ranking import recompute_feed_scores

    async with _session_ctx() as session:
        try:
            solo = await _add_item(session, "Solo-D", _uniq("feed-d.com"), ortho_emb)

            updated = await recompute_feed_scores(session)
            assert updated >= 1

            solo_after = await _get_item(session, solo.id)
            assert solo_after.feed_score is not None
        finally:
            await session.rollback()
