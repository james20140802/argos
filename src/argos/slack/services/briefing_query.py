from __future__ import annotations

import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus, UserAsset

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

_DOMAIN_CAP = 2


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _kmeans(vecs: list[np.ndarray], k: int, max_iter: int = 20, seed: int | None = None) -> list[np.ndarray]:
    data = np.stack(vecs)  # (n, d)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=k, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iter):
        diffs = data[:, None, :] - centroids[None, :, :]  # (n, k, d)
        dists = np.linalg.norm(diffs, axis=-1)            # (n, k)
        labels = np.argmin(dists, axis=-1)                # (n,)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            # TODO: reseed empty cluster to farthest point for better convergence
            new_centroids[i] = data[mask].mean(axis=0) if mask.any() else centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return [centroids[i] for i in range(k)]


async def _embed_topics(topics: list[str]) -> np.ndarray | None:
    if not topics:
        return None
    try:
        from argos.brain.ollama_client import batch_embed
        vecs = await batch_embed(topics)
        arr = np.array(vecs, dtype=np.float32)
        return arr.mean(axis=0)
    except Exception as exc:
        logger.warning("briefing_query: topic embedding failed (%r) — skipping topic boost", exc)
        return None


async def _keep_centroids(session: AsyncSession) -> list[np.ndarray]:
    stmt = (
        select(TechItem.embedding)
        .join(UserAsset, UserAsset.tech_id == TechItem.id)
        .where(UserAsset.status == AssetStatus.KEEP)
        .where(TechItem.embedding.is_not(None))
    )
    result = await session.execute(stmt)
    rows = result.all()

    vecs: list[np.ndarray] = []
    for (emb,) in rows:
        if emb is not None:
            vecs.append(np.array(emb, dtype=np.float32))

    if not vecs:
        return []

    n = len(vecs)
    k = 1 if n < 3 else (2 if n < 9 else (3 if n < 18 else min(5, n // 6)))

    if k == 1:
        return [np.stack(vecs).mean(axis=0)]
    return _kmeans(vecs, k)


def _score_and_select(
    candidates: list[TechItem],
    topic_vec: np.ndarray | None,
    centroids: list[np.ndarray],
    limit: int,
) -> list[TechItem]:
    has_topic = topic_vec is not None
    has_keeps = bool(centroids)

    if has_topic and has_keeps:
        w_trust, w_topic, w_keep = 0.4, 0.35, 0.25
    elif has_topic:
        w_trust, w_topic, w_keep = 0.65, 0.35, 0.0
    elif has_keeps:
        w_trust, w_topic, w_keep = 0.75, 0.0, 0.25
    else:
        w_trust, w_topic, w_keep = 1.0, 0.0, 0.0

    scored: list[tuple[float, TechItem]] = []
    for item in candidates:
        trust = float(item.trust_score or 0.0)

        topic_score = 0.0
        if has_topic and item.embedding is not None:
            item_vec = np.array(item.embedding, dtype=np.float32)
            topic_score = _cosine_sim(item_vec, topic_vec)

        keep_score = 0.0
        if has_keeps and item.embedding is not None:
            item_vec = np.array(item.embedding, dtype=np.float32)
            keep_score = max(_cosine_sim(item_vec, c) for c in centroids)

        jitter = random.uniform(-0.05, 0.05)
        final = w_trust * trust + w_topic * topic_score + w_keep * keep_score + jitter
        scored.append((final, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    domain_counts: dict[str, int] = {}
    selected: list[TechItem] = []
    for _, item in scored:
        domain = urlsplit(item.source_url or "").netloc or "unknown"
        count = domain_counts.get(domain, 0)
        if count >= _DOMAIN_CAP:
            continue
        domain_counts[domain] = count + 1
        selected.append(item)
        if len(selected) >= limit:
            break

    return selected


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

async def fetch_today_briefing(
    session: AsyncSession,
    *,
    limit_per_category: int = 5,
    now_utc: datetime | None = None,
    topics: list[str] | None = None,
    lookback_days: int = 7,
) -> dict[CategoryType, list[TechItem]]:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    cutoff_utc = now_utc - timedelta(days=lookback_days)

    topic_vec = await _embed_topics(topics or [])
    centroids = await _keep_centroids(session)

    pool_limit = limit_per_category * 3

    result: dict[CategoryType, list[TechItem]] = {
        CategoryType.MAINSTREAM: [],
        CategoryType.ALPHA: [],
    }

    for category in (CategoryType.MAINSTREAM, CategoryType.ALPHA):
        effective_date = func.coalesce(TechItem.published_at, TechItem.created_at)
        stmt = (
            select(TechItem)
            .where(
                TechItem.category == category,
                effective_date >= cutoff_utc,
                effective_date <= now_utc,
                TechItem.briefed_at.is_(None),
            )
            .order_by(
                TechItem.trust_score.desc().nulls_last(),
                effective_date.desc(),
            )
            .limit(pool_limit)
        )
        rows = await session.execute(stmt)
        candidates = list(rows.scalars().all())
        result[category] = _score_and_select(candidates, topic_vec, centroids, limit_per_category)

    return result


async def fetch_user_portfolio(
    session: AsyncSession,
    *,
    category: CategoryType | None = None,
    sort_by: Literal["date", "trust"] = "date",
) -> list[tuple[UserAsset, TechItem]]:
    stmt = (
        select(UserAsset, TechItem)
        .join(TechItem, UserAsset.tech_id == TechItem.id)
        .where(UserAsset.status == AssetStatus.KEEP)
    )
    if category is not None:
        stmt = stmt.where(TechItem.category == category)
    if sort_by == "trust":
        stmt = stmt.order_by(
            TechItem.trust_score.desc().nulls_last(),
            UserAsset.updated_at.desc(),
        )
    else:
        stmt = stmt.order_by(UserAsset.updated_at.desc())
    rows = await session.execute(stmt)
    return [(row[0], row[1]) for row in rows.all()]
