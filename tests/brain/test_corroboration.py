"""ARG-210: corroboration pipeline tests.

Uses the real pgvector DB (Docker container must be running), following the
same NullPool/module-skip pattern as
``tests/slack/test_track_check_signal_match_e2e.py`` (no shared ``db_session``
fixture exists in this codebase's conftest — see that file for precedent).

Embeddings:
  - base_emb    : fixed unit vector, treated as "A"
  - similar_emb : base_emb + tiny noise; cosine sim > threshold
  - ortho_emb   : orthogonal direction; cosine sim ~0 (well below threshold)
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain.corroboration import update_corroboration
from argos.brain.trust import (
    corroboration_score,
    score_rubric,
    source_prior,
    synthesize_trust,
)
from argos.config import settings
from argos.models.tech_item import CategoryType, TechItem
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip(
            "pgvector DB not reachable — skipping ARG-210 corroboration tests "
            "(start the Docker DB to run them)"
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
    return float(np.dot(np.array(a), np.array(b)))


@pytest.fixture(scope="module")
def embeddings():
    rng = np.random.default_rng(7)
    base_raw = rng.standard_normal(768)
    similar_raw = base_raw + rng.standard_normal(768) * 0.02
    ortho_raw = rng.standard_normal(768)
    base_unit = _unit(base_raw)
    ortho_raw = ortho_raw - np.dot(ortho_raw, base_unit) * base_unit

    base_emb = _make_emb(base_raw)
    similar_emb = _make_emb(similar_raw)
    ortho_emb = _make_emb(ortho_raw)

    sim_s = _cosine_sim(base_emb, similar_emb)
    sim_o = _cosine_sim(base_emb, ortho_emb)
    assert sim_s > 0.85, f"similar_emb cosine sim too low: {sim_s:.4f}"
    assert sim_o <= 0.85, f"ortho_emb cosine sim too high: {sim_o:.4f}"

    return base_emb, similar_emb, ortho_emb


async def _add_item(
    session,
    title,
    url,
    emb,
    *,
    corroboration_count=None,
    trust_rubric=None,
    trust_score=0.5,
):
    item = TechItem(
        title=title,
        source_url=url,
        raw_content=f"content for {title}",
        category=CategoryType.ALPHA,
        trust_score=trust_score,
        trust_rubric=trust_rubric,
        corroboration_count=corroboration_count,
    )
    session.add(item)
    await session.flush()
    await session.execute(
        text("UPDATE tech_items SET embedding = CAST(:emb AS vector) WHERE id = :id"),
        {"emb": _emb_str(emb), "id": str(item.id)},
    )
    return item


async def _get_item(session, item_id):
    # update_corroboration writes via raw SQL text() UPDATEs, which bypass the
    # ORM identity map. With expire_on_commit=False the already-loaded `a`
    # instance would otherwise keep serving stale in-memory column values, so
    # force a fresh read from the DB.
    result = await session.execute(
        select(TechItem)
        .where(TechItem.id == item_id)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one()


def _uniq(tag: str) -> str:
    return f"https://{tag}/{uuid.uuid4()}"


@pytest.mark.asyncio
async def test_corroboration_counts_distinct_domains_only(embeddings):
    """A(a.com) is similar to B(b.com) and C(a.com); only B (different domain)
    should count. corroboration_count for A must be 1."""
    base_emb, similar_emb, _ortho_emb = embeddings
    async with _session_ctx() as session:
        try:
            a = await _add_item(session, "A", _uniq("a.com"), base_emb)
            await _add_item(session, "B", _uniq("b.com"), similar_emb)
            await _add_item(session, "C", _uniq("a.com"), similar_emb)

            await update_corroboration(session)

            a_after = await _get_item(session, a.id)
            assert a_after.corroboration_count == 1
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_corroboration_zero_when_no_similar_items(embeddings):
    """An item with only an orthogonal (dissimilar) neighbor gets count 0."""
    base_emb, _similar_emb, ortho_emb = embeddings
    async with _session_ctx() as session:
        try:
            a = await _add_item(session, "A-solo", _uniq("solo-a.com"), base_emb)
            await _add_item(session, "D-ortho", _uniq("solo-d.com"), ortho_emb)

            await update_corroboration(session)

            a_after = await _get_item(session, a.id)
            assert (a_after.corroboration_count or 0) == 0
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_resynthesizes_trust_score_when_rubric_present(embeddings):
    """Item with a trust_rubric gets trust_score recomputed to match
    synthesize_trust(score_rubric, source_prior, corroboration_score, weights)
    once its corroboration_count changes."""
    base_emb, similar_emb, _ortho_emb = embeddings
    rubric = {
        "is_primary_source": True,
        "has_evidence_links": True,
        "has_concrete_numbers": True,
        "claim_evidence_balance": "balanced",
        "marketing_intensity": "low",
    }
    a_url = _uniq("rubric-a.com")
    async with _session_ctx() as session:
        try:
            a = await _add_item(
                session,
                "A-rubric",
                a_url,
                base_emb,
                trust_rubric=rubric,
                trust_score=0.1,  # stale value, must be overwritten
            )
            await _add_item(session, "B-rubric", _uniq("rubric-b.com"), similar_emb)

            await update_corroboration(session)

            a_after = await _get_item(session, a.id)
            assert a_after.corroboration_count == 1

            trust_cfg = settings.user.trust
            weights = {
                "rubric": trust_cfg.weight_rubric,
                "prior": trust_cfg.weight_prior,
                "corroboration": trust_cfg.weight_corroboration,
            }
            expected = synthesize_trust(
                score_rubric(rubric),
                source_prior(a_url, trust_cfg.source_tiers),
                corroboration_score(1),
                weights,
            )
            assert a_after.trust_score == pytest.approx(expected)
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_legacy_row_without_rubric_keeps_trust_score(embeddings):
    """Item with trust_rubric=NULL only gets corroboration_count updated;
    trust_score (legacy LLM-assigned value) must be left untouched."""
    base_emb, similar_emb, _ortho_emb = embeddings
    async with _session_ctx() as session:
        try:
            a = await _add_item(
                session,
                "A-legacy",
                _uniq("legacy-a.com"),
                base_emb,
                trust_rubric=None,
                trust_score=0.77,
            )
            await _add_item(session, "B-legacy", _uniq("legacy-b.com"), similar_emb)

            await update_corroboration(session)

            a_after = await _get_item(session, a.id)
            assert a_after.corroboration_count == 1
            assert a_after.trust_score == pytest.approx(0.77)
        finally:
            await session.rollback()


@pytest.mark.asyncio
async def test_threshold_config_changes_count(embeddings, monkeypatch):
    """Lowering corroboration_threshold below a borderline similarity should
    make items count as corroborating that a stricter threshold would not."""
    base_emb, similar_emb, _ortho_emb = embeddings
    sim = _cosine_sim(base_emb, similar_emb)

    async with _session_ctx() as session:
        try:
            a = await _add_item(session, "A-thresh", _uniq("thresh-a.com"), base_emb)
            await _add_item(session, "B-thresh", _uniq("thresh-b.com"), similar_emb)

            # Threshold above the actual similarity: must NOT count. Midpoint
            # between sim and 1.0 is always strictly greater than sim (sim<1),
            # unlike a fixed additive offset which can overshoot past 1.0 and
            # get clamped back below sim when sim is already very close to 1.
            high_threshold = (sim + 1.0) / 2
            monkeypatch.setattr(
                settings.user.trust, "corroboration_threshold", high_threshold
            )
            await update_corroboration(session)
            a_high = await _get_item(session, a.id)
            assert (a_high.corroboration_count or 0) == 0

            # Threshold below the actual similarity: must count.
            monkeypatch.setattr(settings.user.trust, "corroboration_threshold", 0.5)
            await update_corroboration(session)
            a_low = await _get_item(session, a.id)
            assert a_low.corroboration_count == 1
        finally:
            await session.rollback()
