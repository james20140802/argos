"""Read-side service backing the 상세 보기 screen (ARG-158, ARG-159).

``fetch_item_detail`` returns a single ``tech_item`` enriched with the
fields needed to render the in-app reader:

* T1 (ARG-158): hero image, title, trust-score dial, summary, source link.
* T2 (ARG-159): 🧬 genealogy — predecessors + successors with
  ``relation_type`` + ``reasoning``.

Subsequent slices will extend ``ItemDetailView`` with pgvector similarity
(ARG-160) and track_history (ARG-161) without changing the existing
contract.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import RelationType, TechSuccession


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
    )
