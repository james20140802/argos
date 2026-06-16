"""Read-side service backing the 상세 보기 screen (ARG-158/159/160).

``fetch_item_detail`` returns a single ``tech_item`` enriched with the
fields needed to render the in-app reader:

* T1 (ARG-158): hero image, title, trust-score dial, summary, source link.
* T2 (ARG-159): 🧬 genealogy — predecessors + successors with
  ``relation_type`` + ``reasoning``.
* T4 (ARG-160): 🔭 related signals — pgvector top-5 similarity vs Keep
  user_assets (excluding current item id).

The remaining slice (ARG-161) will extend ``ItemDetailView`` with
track_history without changing the existing contract.
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


# Number of pgvector-similar items shown in the 관련 신호 → similarity
# subsection (ARG-160).
SIMILAR_LIMIT: int = 5


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

    ``tech_id`` / ``title`` describe the recommended tech item; the
    cosine distance to the closest Keep asset is not exposed — the
    view is a flat top-K list.
    """

    tech_id: uuid.UUID
    title: str


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
    """Top-K tech_items closest (cosine `<=>`) to any Keep user_asset's embedding.

    The current item is excluded. Items without an embedding are skipped on
    both sides of the comparison. Result is empty when no Keep asset exists,
    no embeddings are present, or there is no candidate other than the
    anchor itself.
    """
    sql = text(
        "SELECT t.id, t.title, MIN(t.embedding <=> k.embedding) AS dist "
        "FROM tech_items t "
        "CROSS JOIN tech_items k "
        "JOIN user_assets ua ON ua.tech_id = k.id AND ua.status = 'Keep' "
        "WHERE t.id <> :item_id "
        "  AND t.embedding IS NOT NULL "
        "  AND k.embedding IS NOT NULL "
        "GROUP BY t.id, t.title "
        "ORDER BY dist ASC "
        "LIMIT :limit"
    )
    rows = (
        await session.execute(sql, {"item_id": str(item_id), "limit": limit})
    ).fetchall()
    return [SimilarItem(tech_id=row.id, title=row.title) for row in rows]


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
    )
