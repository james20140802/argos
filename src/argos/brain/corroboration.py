"""Corroboration pipeline (ARG-210).

Every ``argos run`` end, count distinct-domain similar items (pgvector
cosine similarity above ``corroboration_threshold``) over the recent
``corroboration_lookback_days`` window into ``tech_items.corroboration_count``,
and re-synthesize ``trust_score`` (ARG-206) for rows whose count changed.

Same-domain "echoes" (e.g. two posts on the same blog) never count toward
corroboration — only independent sources do. Legacy rows without a
``trust_rubric`` (pre-ARG-206 LLM-assigned trust_score) only get their
``corroboration_count`` refreshed; their ``trust_score`` is left untouched
since it cannot be re-synthesized without the rubric.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.trust import (
    corroboration_score,
    score_rubric,
    source_prior,
    synthesize_trust,
)
from argos.config import settings


def _netloc(url: str | None) -> str:
    """Lower-cased, www.-stripped domain — mirrors argos.brain.trust.source_prior."""
    netloc = urlparse(url or "").netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[len("www."):]
    return netloc


async def update_corroboration(session: AsyncSession) -> int:
    """Recompute distinct-domain corroboration counts for recent tech_items.

    For every item created within ``settings.user.trust.corroboration_lookback_days``
    days, counts the number of *other* tech_items (any age) whose embedding
    cosine similarity to it exceeds ``settings.user.trust.corroboration_threshold``
    AND whose source domain differs from the item's own domain.

    Rows whose count changed are UPDATEd. If the row also has a non-NULL
    ``trust_rubric``, ``trust_score`` is re-synthesized in the same UPDATE via
    ``synthesize_trust(score_rubric(rubric), source_prior(url, tiers),
    corroboration_score(new_count), weights)``. Legacy rows (``trust_rubric``
    IS NULL) only get ``corroboration_count`` updated — ``trust_score`` is
    left untouched.

    Does not commit; the caller (``argos run`` end hook) commits.

    Returns the number of rows updated.
    """
    trust_cfg = settings.user.trust
    threshold = trust_cfg.corroboration_threshold
    lookback_days = trust_cfg.corroboration_lookback_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    recent_rows = (
        await session.execute(
            text(
                """
                SELECT id, source_url, corroboration_count, trust_rubric
                FROM tech_items
                WHERE created_at >= :cutoff
                  AND embedding IS NOT NULL
                """
            ),
            {"cutoff": cutoff},
        )
    ).mappings().all()

    if not recent_rows:
        return 0

    # Candidate similar pairs for the recent set. Domain-distinctness is
    # cheaper and clearer to resolve in Python than in SQL (see brief) — the
    # recent window is small, so this post-processing is fine. Both sides of
    # the pair are constrained to the lookback window: corroboration means
    # "how many recent independent sources agree", so an out-of-window (old)
    # neighbour must not inflate a recent item's count.
    pair_rows = (
        await session.execute(
            text(
                """
                SELECT a.id AS item_id, b.source_url AS other_url
                FROM tech_items a
                JOIN tech_items b ON b.id != a.id
                WHERE a.created_at >= :cutoff
                  AND b.created_at >= :cutoff
                  AND a.embedding IS NOT NULL
                  AND b.embedding IS NOT NULL
                  AND (1.0 - (a.embedding <=> b.embedding)) > :threshold
                """
            ),
            {"cutoff": cutoff, "threshold": threshold},
        )
    ).mappings().all()

    domains_by_item: dict = {}
    for row in pair_rows:
        domains_by_item.setdefault(row["item_id"], set()).add(_netloc(row["other_url"]))

    weights = {
        "rubric": trust_cfg.weight_rubric,
        "prior": trust_cfg.weight_prior,
        "corroboration": trust_cfg.weight_corroboration,
    }
    tiers = trust_cfg.source_tiers

    updated = 0
    for row in recent_rows:
        item_id = row["id"]
        item_url = row["source_url"]
        item_netloc = _netloc(item_url)

        other_domains = domains_by_item.get(item_id, set())
        other_domains.discard(item_netloc)
        new_count = len(other_domains)
        old_count = row["corroboration_count"] or 0

        if new_count == old_count:
            continue

        rubric = row["trust_rubric"]
        if rubric is not None:
            trust_score = synthesize_trust(
                score_rubric(rubric),
                source_prior(item_url, tiers),
                corroboration_score(new_count),
                weights,
            )
            await session.execute(
                text(
                    """
                    UPDATE tech_items
                    SET corroboration_count = :count, trust_score = :trust_score
                    WHERE id = :id
                    """
                ),
                {"count": new_count, "trust_score": trust_score, "id": item_id},
            )
        else:
            await session.execute(
                text(
                    """
                    UPDATE tech_items
                    SET corroboration_count = :count
                    WHERE id = :id
                    """
                ),
                {"count": new_count, "id": item_id},
            )
        updated += 1

    return updated
