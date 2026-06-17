"""Read-side service backing the 상세 보기 screen (ARG-158/159/160/161).

``fetch_item_detail`` returns a single ``tech_item`` enriched with the
fields needed to render the in-app reader:

* T1 (ARG-158): hero image, title, trust-score dial, summary, source link.
* T2 (ARG-159): 🧬 genealogy — predecessors + successors with
  ``relation_type`` + ``reasoning``.
* T4 (ARG-160): 🔭 related signals — pgvector top-5 Keep user_assets
  ranked by similarity to the current item (excluding current item id).
* T3 (ARG-161): 🔭 related signals — 10 most recent track_history rows
  scoped to user_assets tied to the current item OR similarity tech ids.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import RelationType, TechSuccession
from argos.models.track_history import TrackHistory
from argos.models.user_asset import UserAsset


# Number of pgvector-similar items shown in the 관련 신호 → similarity
# subsection (ARG-160).
SIMILAR_LIMIT: int = 5

# Number of track_history rows shown in the 관련 신호 → timeline
# subsection (ARG-161).
HISTORY_LIMIT: int = 10


@dataclass(frozen=True)
class GenealogyEntry:
    """One predecessor or successor of the current item.

    ``tech_id`` / ``title`` describe the OTHER tech item; ``relation_type``
    and ``reasoning`` come straight from ``tech_succession``.
    """

    tech_id: uuid.UUID
    title: str
    relation_type: RelationType
    reasoning: Optional[str]


@dataclass(frozen=True)
class SimilarItem:
    """One pgvector-similar tech item for the 🔭 관련 신호 subsection.

    ``tech_id`` / ``title`` describe the recommended Keep asset; the
    cosine distance to the current item is not exposed — the view is a
    flat top-K list.
    """

    tech_id: uuid.UUID
    title: str


@dataclass(frozen=True)
class HistoryEntry:
    """One row of the 🔭 관련 신호 → track_history timeline (ARG-161).

    ``tech_title`` is the title of the tech_item the user_asset points
    at, so the timeline reads as "TechX: Tracking → Keep at …" rather
    than referring back to an opaque user_asset id.
    """

    changed_from: str
    changed_to: str
    changed_at: datetime
    tech_id: uuid.UUID
    tech_title: str


@dataclass(frozen=True)
class ItemDetailView:
    id: uuid.UUID
    title: str
    source_url: str
    image_url: Optional[str]
    summary: Optional[str]
    category: Optional[CategoryType]
    trust_score: Optional[float]
    published_at: Optional[datetime]
    predecessors: list[GenealogyEntry] = field(default_factory=list)
    successors: list[GenealogyEntry] = field(default_factory=list)
    similar: list[SimilarItem] = field(default_factory=list)
    related_history: list[HistoryEntry] = field(default_factory=list)


async def _fetch_predecessors(
    session: AsyncSession, item_id: uuid.UUID
) -> list[GenealogyEntry]:
    """Items that came BEFORE the current item — joined via predecessor_id."""
    Pred = aliased(TechItem)
    stmt = (
        select(
            Pred.id,
            Pred.title,
            TechSuccession.relation_type,
            TechSuccession.reasoning,
        )
        .join(Pred, Pred.id == TechSuccession.predecessor_id)
        .where(TechSuccession.successor_id == item_id)
        .order_by(TechSuccession.created_at.asc())
    )
    rows = (await session.execute(stmt)).all()
    return [
        GenealogyEntry(
            tech_id=row.id,
            title=row.title,
            relation_type=row.relation_type,
            reasoning=row.reasoning,
        )
        for row in rows
    ]


async def _fetch_successors(
    session: AsyncSession, item_id: uuid.UUID
) -> list[GenealogyEntry]:
    """Items that came AFTER the current item — joined via successor_id."""
    Succ = aliased(TechItem)
    stmt = (
        select(
            Succ.id,
            Succ.title,
            TechSuccession.relation_type,
            TechSuccession.reasoning,
        )
        .join(Succ, Succ.id == TechSuccession.successor_id)
        .where(TechSuccession.predecessor_id == item_id)
        .order_by(TechSuccession.created_at.asc())
    )
    rows = (await session.execute(stmt)).all()
    return [
        GenealogyEntry(
            tech_id=row.id,
            title=row.title,
            relation_type=row.relation_type,
            reasoning=row.reasoning,
        )
        for row in rows
    ]


async def _fetch_similar(
    session: AsyncSession,
    item_id: uuid.UUID,
    limit: int = SIMILAR_LIMIT,
) -> list[SimilarItem]:
    """Top-K Keep assets closest (cosine `<=>`) to the *current item*'s embedding.

    The vector comparison is anchored on the item being viewed: candidates are
    the user's Keep assets, ranked by their cosine distance to this item, so
    ``관련 신호`` surfaces the tracked assets most related to what's on screen.
    The current item is excluded. Result is empty when the current item has no
    embedding, no Keep asset exists, or no Keep asset has an embedding.
    """
    sql = text(
        "SELECT t.id, t.title, MIN(t.embedding <=> c.embedding) AS dist "
        "FROM tech_items t "
        "JOIN user_assets ua ON ua.tech_id = t.id AND ua.status = 'Keep' "
        "CROSS JOIN (SELECT embedding FROM tech_items WHERE id = :item_id) c "
        "WHERE t.id <> :item_id "
        "  AND t.embedding IS NOT NULL "
        "  AND c.embedding IS NOT NULL "
        "GROUP BY t.id, t.title "
        "ORDER BY dist ASC "
        "LIMIT :limit"
    )
    rows = (
        await session.execute(sql, {"item_id": str(item_id), "limit": limit})
    ).fetchall()
    return [SimilarItem(tech_id=row.id, title=row.title) for row in rows]


async def _fetch_related_history(
    session: AsyncSession,
    tech_ids: list[uuid.UUID],
    limit: int = HISTORY_LIMIT,
) -> list[HistoryEntry]:
    """Most recent ``track_history`` rows for user_assets in ``tech_ids``.

    Empty when no user_asset exists for any of those tech_ids.
    """
    if not tech_ids:
        return []
    stmt = (
        select(
            TrackHistory.changed_from,
            TrackHistory.changed_to,
            TrackHistory.changed_at,
            TechItem.id.label("tech_id"),
            TechItem.title.label("tech_title"),
        )
        .join(UserAsset, UserAsset.id == TrackHistory.user_asset_id)
        .join(TechItem, TechItem.id == UserAsset.tech_id)
        .where(TechItem.id.in_(tech_ids))
        .order_by(TrackHistory.changed_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        HistoryEntry(
            changed_from=row.changed_from,
            changed_to=row.changed_to,
            changed_at=row.changed_at,
            tech_id=row.tech_id,
            tech_title=row.tech_title,
        )
        for row in rows
    ]


async def fetch_item_detail(
    session: AsyncSession,
    item_id: uuid.UUID,
) -> Optional[ItemDetailView]:
    """Return the detail view for ``item_id`` or ``None`` when unknown."""
    stmt = select(
        TechItem.id,
        TechItem.title,
        TechItem.source_url,
        TechItem.image_url,
        TechItem.summary,
        TechItem.category,
        TechItem.trust_score,
        TechItem.published_at,
    ).where(TechItem.id == item_id)

    row = (await session.execute(stmt)).first()
    if row is None:
        return None

    predecessors = await _fetch_predecessors(session, item_id)
    successors = await _fetch_successors(session, item_id)
    similar = await _fetch_similar(session, item_id)
    related_history = await _fetch_related_history(
        session,
        [item_id] + [s.tech_id for s in similar],
    )

    return ItemDetailView(
        id=row.id,
        title=row.title,
        source_url=row.source_url,
        image_url=row.image_url,
        summary=row.summary,
        category=row.category,
        trust_score=row.trust_score,
        published_at=row.published_at,
        predecessors=predecessors,
        successors=successors,
        similar=similar,
        related_history=related_history,
    )
