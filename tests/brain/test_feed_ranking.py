"""ARG-212: feed_score batch ranking tests.

Pure scoring-function tests are DB-free. The batch `recompute_feed_scores`
test uses the real pgvector DB, following the same NullPool/module-skip
pattern as ``tests/brain/test_corroboration.py``.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

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


def test_profile_recency_confidence_fades_stale_keeps():
    from argos.brain.feed_ranking import profile_recency_confidence

    now = datetime.now(timezone.utc)
    hl = 48.0
    old_ts = now - timedelta(days=182)
    # No keeps → 0.0 (profile term contributes nothing).
    assert profile_recency_confidence([], now=now, half_life_hours=hl) == 0.0
    # A single fresh Keep → ~1.0 (unchanged behavior).
    fresh = profile_recency_confidence([([1.0], now)], now=now, half_life_hours=hl)
    assert fresh > 0.99
    # A single 6-month-old Keep → ~0.0: the profile term must vanish absolutely,
    # not just re-weight relative to other Keeps.
    stale = profile_recency_confidence([([1.0], old_ts)], now=now, half_life_hours=hl)
    assert stale < 0.01
    # Genuinely old-only Keeps stay faded no matter how MANY there are — the
    # sum of ~0 decays is still ~0 (not lifted by count).
    many_stale = profile_recency_confidence(
        [([1.0], old_ts)] * 100, now=now, half_life_hours=hl
    )
    assert many_stale < 0.01


def test_profile_recency_confidence_preserves_fresh_in_stale_portfolio():
    # P2 fix (Codex review): a fresh Keep must keep its influence even amid a
    # large stale history. A mean (÷ total Keep count) would drown it — 1 fresh
    # + 99 six-month-old → ~0.01. Summing the recent signal mass keeps it ~1.0.
    from argos.brain.feed_ranking import profile_recency_confidence

    now = datetime.now(timezone.utc)
    hl = 48.0
    old_ts = now - timedelta(days=182)
    keeps = [([1.0], now)] + [([1.0], old_ts)] * 99
    conf = profile_recency_confidence(keeps, now=now, half_life_hours=hl)
    assert conf > 0.99  # fresh signal preserved, not diluted to ~1/100


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


async def _set_created_at(session, item, when) -> None:
    await session.execute(
        text("UPDATE tech_items SET created_at = :ts WHERE id = :id"),
        {"ts": when, "id": str(item.id)},
    )


async def _keep_at(session, tech_item, kept_at) -> None:
    """Create a Keep asset whose *interaction time* (when the user Kept it) is
    ``kept_at``.

    The ranker sources Keep recency from the latest track_history Keep-transition,
    falling back to ``UserAsset.created_at`` — deliberately NOT
    ``last_monitored_at``, which an unrelated signal-match notifier also bumps
    (ARG-201). A fresh item Kept directly into Keep state logs no transition row,
    so its creation time IS the Keep time; we stamp ``created_at`` to model that
    common path.
    """
    asset = UserAsset(tech_id=tech_item.id, status=AssetStatus.KEEP)
    session.add(asset)
    await session.flush()
    await session.execute(
        text("UPDATE user_assets SET created_at = :ts WHERE id = :id"),
        {"ts": kept_at, "id": str(asset.id)},
    )


@pytest.mark.asyncio
async def test_recompute_weights_keep_by_interaction_time_not_crawl_time():
    """P2 regression (Codex review): the Keep signal must decay from *when the
    user Kept it*, not the item's crawl time (TechItem.created_at).

    Two Keeps pull the profile in orthogonal directions X and Y. Keep-X was
    crawled recently; Keep-Y was crawled a year ago but Kept *today*. Under the
    ~48h half-life, weighting Y by its ancient TechItem.created_at decays it to
    ~0, so the profile collapses onto X alone and a Y-similar candidate gets no
    boost. Weighting by the Keep interaction time keeps Y live (profile ≈
    (X+Y)/2), so the Y-similar candidate's profile cosine jumps from ~0 to
    ~0.707.

    Both candidates share trust/corroboration/recency, so the only score
    differentiator is the profile term (weight_profile). We assert a margin
    that only the live-Y profile contribution can clear: without the fix the
    two scores are ~equal (Δ≈0) and the margin assertion fails."""
    from argos.brain.feed_ranking import recompute_feed_scores

    cfg = settings.user.feed_ranking
    # With the fix, Δfeed_score ≈ weight_profile * 0.707; pick a margin well
    # below that but far above the fixless Δ≈0 (float noise only).
    margin = 0.4 * cfg.weight_profile

    rng = np.random.default_rng(29)
    x = _unit(rng.standard_normal(768))
    y0 = rng.standard_normal(768)
    y = _unit(y0 - np.dot(y0, x) * x)  # orthogonal to x
    cand_y = _make_emb(y + rng.standard_normal(768) * 0.02)  # ~similar to y
    z0 = rng.standard_normal(768)
    z = z0 - np.dot(z0, x) * x
    z = _make_emb(z - np.dot(z, y) * y)  # orthogonal to both x and y

    now = datetime.now(timezone.utc)
    async with _session_ctx() as session:
        try:
            keep_x = await _add_item(session, "Keep-X", _uniq("kx.com"), _make_emb(x))
            await _keep_at(session, keep_x, now)

            keep_y = await _add_item(session, "Keep-Y-old", _uniq("ky.com"), _make_emb(y))
            await _set_created_at(session, keep_y, now - timedelta(days=365))
            await _keep_at(session, keep_y, now)  # Kept today despite old crawl

            cand = await _add_item(
                session, "Cand-Y", _uniq("cy.com"), cand_y,
                trust_score=0.4, corroboration_count=2,
            )
            other = await _add_item(
                session, "Cand-Z", _uniq("cz.com"), z,
                trust_score=0.4, corroboration_count=2,
            )

            await recompute_feed_scores(session)

            cand_after = await _get_item(session, cand.id)
            other_after = await _get_item(session, other.id)
            assert cand_after.feed_score is not None
            assert other_after.feed_score is not None
            # Y stays in the profile only if weighted by interaction time; the
            # margin is unreachable when Y decays away under the old crawl-time
            # weighting.
            assert cand_after.feed_score > other_after.feed_score + margin
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_recompute_stale_only_profile_fades_absolutely(embeddings):
    """P2 regression (Codex review): a profile built from an all-stale Keep set
    must fade *absolutely* — the mean-normalized profile direction is magnitude-
    free, so without the profile_recency_confidence scaling a six-month-old sole
    Keep would boost matches as strongly as a fresh one.

    One Keep, Kept 6 months ago. A base-aligned candidate and an orthogonal one
    share trust/corroboration/recency. With the confidence scaling the profile
    term ~vanishes, so the two land within a tight band; without it the aligned
    candidate would win by ≈weight_profile·cos (~0.34). Assert the gap stays
    small — an assertion the unscaled profile term cannot satisfy."""
    base_emb, similar_emb, ortho_emb = embeddings
    from argos.brain.feed_ranking import recompute_feed_scores

    now = datetime.now(timezone.utc)
    cfg = settings.user.feed_ranking
    # Unscaled gap would be ≈weight_profile*cos(similar,base) (~0.34); the scaled
    # gap is ≈that * confidence(6mo) ≈ 0. Pick a band well below the unscaled gap.
    band = 0.25 * cfg.weight_profile

    async with _session_ctx() as session:
        try:
            kept = await _add_item(session, "Kept-old", _uniq("old.com"), base_emb)
            # Kept 6 months ago and untouched since → stale interaction time.
            await _keep_at(session, kept, now - timedelta(days=182))

            aligned = await _add_item(
                session, "Aligned", _uniq("al.com"), similar_emb,
                trust_score=0.4, corroboration_count=2,
            )
            ortho = await _add_item(
                session, "Ortho", _uniq("or.com"), ortho_emb,
                trust_score=0.4, corroboration_count=2,
            )

            await recompute_feed_scores(session)

            a_after = await _get_item(session, aligned.id)
            o_after = await _get_item(session, ortho.id)
            assert a_after.feed_score is not None
            assert o_after.feed_score is not None
            # Stale-only profile: aligned must NOT out-boost orthogonal — the
            # faded profile term keeps them within a tight band.
            assert abs(a_after.feed_score - o_after.feed_score) < band
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_recompute_keep_recency_ignores_signal_match_bump(embeddings):
    """ARG-201 regression: an automated signal-match Slack alert bumps
    ``UserAsset.last_monitored_at`` to now() with NO user action
    (``track_check.post_signal_update``). The Keep recency must decay from *when
    the user actually Kept it* (track_history Keep-transition, fallback
    ``UserAsset.created_at``), never from ``last_monitored_at`` — otherwise a
    long-abandoned Keep is silently re-affirmed and its profile influence never
    fades, the opposite of ``profile_recency_confidence``'s promise.

    A sole Keep, Kept 6 months ago and untouched by the user since, but whose
    ``last_monitored_at`` was refreshed to now by a signal match, must still fade
    absolutely — landing the aligned and orthogonal candidates within the same
    tight band as the untouched-stale case. Under the old ``last_monitored_at``
    weighting the profile would read as fresh and the aligned candidate would win
    by ≈``weight_profile·cos``, blowing past the band."""
    base_emb, similar_emb, ortho_emb = embeddings
    from argos.brain.feed_ranking import recompute_feed_scores

    now = datetime.now(timezone.utc)
    cfg = settings.user.feed_ranking
    band = 0.25 * cfg.weight_profile

    async with _session_ctx() as session:
        try:
            kept = await _add_item(
                session, "Kept-old-signalled", _uniq("os.com"), base_emb
            )
            asset = UserAsset(tech_id=kept.id, status=AssetStatus.KEEP)
            session.add(asset)
            await session.flush()
            # Kept 6 months ago (created_at), untouched by the user since...
            await session.execute(
                text("UPDATE user_assets SET created_at = :ts WHERE id = :id"),
                {"ts": now - timedelta(days=182), "id": str(asset.id)},
            )
            # ...but an automated signal match today bumped last_monitored_at
            # with no user interaction — this must NOT re-affirm the Keep.
            await session.execute(
                text("UPDATE user_assets SET last_monitored_at = :ts WHERE id = :id"),
                {"ts": now, "id": str(asset.id)},
            )

            aligned = await _add_item(
                session, "Aligned", _uniq("al2.com"), similar_emb,
                trust_score=0.4, corroboration_count=2,
            )
            ortho = await _add_item(
                session, "Ortho", _uniq("or2.com"), ortho_emb,
                trust_score=0.4, corroboration_count=2,
            )

            await recompute_feed_scores(session)

            a_after = await _get_item(session, aligned.id)
            o_after = await _get_item(session, ortho.id)
            assert a_after.feed_score is not None
            assert o_after.feed_score is not None
            # The signal-match last_monitored_at=now must not refresh the stale
            # Keep: the faded profile term keeps the two within a tight band.
            assert abs(a_after.feed_score - o_after.feed_score) < band
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
