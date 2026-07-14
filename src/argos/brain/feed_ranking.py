"""Recommendation feed ranking (ARG-212).

Every ``argos run`` end (after corroboration has refreshed
``corroboration_count``), recompute ``tech_items.feed_score`` for every row
from four weighted terms:

- **recency**: exponential decay of ``now - (published_at or created_at)``.
- **profile**: cosine similarity between the item's embedding and a
  user-profile vector built from Keep-ed embeddings (time-decayed weighted
  mean) minus a Pass-ed (Archived) embeddings mean, scaled by
  ``pass_weight``. 0.0 when there's no profile yet (cold start) or the item
  has no embedding.
- **trust**: ``trust_score`` (0.0 when NULL).
- **trending**: corroboration reuse — ``trending_score(corroboration_count)``.

Plus a small additive ``interest_bonus`` when the item's title/summary
contains one of ``interests.topics`` (substring match, deliberately simple).

Mirrors ``argos.brain.corroboration.update_corroboration``'s shape: a
brain-side batch that reads via the ORM/raw SQL, UPDATEs rows, and does NOT
commit — the caller (``argos run`` end hook) commits.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.config import settings
from argos.models.tech_item import TechItem
from argos.models.user_asset import AssetStatus, UserAsset


def recency_decay(age_hours: float, half_life_hours: float) -> float:
    """Exponential half-life decay: ``0.5 ** (age_hours / half_life_hours)``.

    Negative ``age_hours`` (e.g. a ``published_at`` slightly in the future due
    to clock skew) is clamped to 0.0 → full-strength 1.0. A non-positive
    ``half_life_hours`` decays instantly to 0.0 except at age 0.
    """
    if age_hours < 0:
        age_hours = 0.0
    if half_life_hours <= 0:
        return 1.0 if age_hours == 0 else 0.0
    return 0.5 ** (age_hours / half_life_hours)


def trending_score(count: int | None) -> float:
    """Monotonic, (0, 1)-bounded transform of corroboration count.

    ``count / (count + 1)``; ``None`` or negative counts are treated as 0.
    """
    c = count or 0
    if c < 0:
        c = 0
    return c / (c + 1)


def compute_profile_vector(
    keep_embeds_with_ts: list[tuple[list[float], datetime]],
    pass_embeds: list[list[float]],
    *,
    now: datetime,
    half_life_hours: float,
    pass_weight: float,
) -> list[float] | None:
    """Build the user-profile vector: Keep time-decayed mean − pass_weight * Pass mean.

    Returns ``None`` when there are no Keep embeddings (cold start — nothing
    to build a profile from). Pass embeddings without a Keep signal have no
    meaningful "direction" to subtract from, so they alone don't produce a
    profile either.
    """
    if not keep_embeds_with_ts:
        return None

    dim = len(keep_embeds_with_ts[0][0])
    weighted_sum = [0.0] * dim
    weight_total = 0.0
    for embed, kept_at in keep_embeds_with_ts:
        age_hours = (now - kept_at).total_seconds() / 3600.0
        weight = recency_decay(age_hours, half_life_hours)
        weight_total += weight
        for i, value in enumerate(embed):
            weighted_sum[i] += weight * value

    if weight_total > 0:
        keep_mean = [value / weight_total for value in weighted_sum]
    else:
        keep_mean = [0.0] * dim

    if pass_embeds:
        pass_sum = [0.0] * len(pass_embeds[0])
        for embed in pass_embeds:
            for i, value in enumerate(embed):
                pass_sum[i] += value
        pass_mean = [value / len(pass_embeds) for value in pass_sum]
    else:
        pass_mean = [0.0] * dim

    return [k - pass_weight * p for k, p in zip(keep_mean, pass_mean)]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def recompute_feed_scores(session: AsyncSession) -> int:
    """Recompute ``feed_score`` for every tech_item.

    Builds the user-profile vector from current Keep/Pass(=Archived) asset
    embeddings, then scores every row as the config-weighted sum of recency
    decay, profile cosine similarity, trust, and trending, plus a small
    interest-topic bonus.

    Does not commit; the caller (``argos run`` end hook) commits.

    Returns the number of rows updated.
    """
    cfg = settings.user.feed_ranking
    now = datetime.now(timezone.utc)

    # Weight each Keep by *when the user Kept it*, not when the item was
    # crawled. transition_asset stamps UserAsset.last_monitored_at with the
    # interaction time, so a months-old item Kept today is a fresh signal;
    # using TechItem.created_at would decay it to ~0 under the recency
    # half-life and under-rank its similar recommendations. Fall back to the
    # crawl time when last_monitored_at is somehow NULL (nullable column).
    keep_ts = func.coalesce(UserAsset.last_monitored_at, TechItem.created_at)
    keep_rows = (
        await session.execute(
            select(TechItem.embedding, keep_ts)
            .join(UserAsset, UserAsset.tech_id == TechItem.id)
            .where(UserAsset.status == AssetStatus.KEEP)
            .where(TechItem.embedding.is_not(None))
        )
    ).all()
    pass_rows = (
        await session.execute(
            select(TechItem.embedding)
            .join(UserAsset, UserAsset.tech_id == TechItem.id)
            .where(UserAsset.status == AssetStatus.ARCHIVED)
            .where(TechItem.embedding.is_not(None))
        )
    ).all()

    keep = [(list(emb), kept_at) for emb, kept_at in keep_rows if emb is not None]
    passv = [list(emb) for (emb,) in pass_rows if emb is not None]

    profile = compute_profile_vector(
        keep,
        passv,
        now=now,
        half_life_hours=cfg.recency_half_life_hours,
        pass_weight=cfg.pass_weight,
    )
    profile_arr = np.array(profile, dtype=np.float32) if profile is not None else None

    topics = [t.lower() for t in settings.user.interests.topics if t]

    rows = (
        await session.execute(
            select(
                TechItem.id,
                TechItem.embedding,
                TechItem.trust_score,
                TechItem.corroboration_count,
                TechItem.published_at,
                TechItem.created_at,
                TechItem.title,
                TechItem.summary,
            )
        )
    ).all()

    updated = 0
    for row in rows:
        (
            item_id,
            embedding,
            trust_score,
            corroboration_count,
            published_at,
            created_at,
            title,
            summary,
        ) = row

        ts = published_at or created_at
        age_hours = (now - ts).total_seconds() / 3600.0 if ts is not None else 0.0
        recency = recency_decay(age_hours, cfg.recency_half_life_hours)

        profile_sim = 0.0
        if profile_arr is not None and embedding is not None:
            item_vec = np.array(embedding, dtype=np.float32)
            profile_sim = _cosine_sim(item_vec, profile_arr)

        trust = float(trust_score or 0.0)
        trending = trending_score(corroboration_count)

        interest_bonus = 0.0
        if topics:
            haystack = f"{title or ''} {summary or ''}".lower()
            if any(topic in haystack for topic in topics):
                interest_bonus = cfg.interest_bonus

        score = (
            cfg.weight_recency * recency
            + cfg.weight_profile * profile_sim
            + cfg.weight_trust * trust
            + cfg.weight_trending * trending
            + interest_bonus
        )

        await session.execute(
            text("UPDATE tech_items SET feed_score = :score WHERE id = :id"),
            {"score": score, "id": item_id},
        )
        updated += 1

    return updated
