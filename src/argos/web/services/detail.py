"""Read-side service backing the 상세 보기 screen (ARG-158).

``fetch_item_detail`` returns a single ``tech_item`` enriched with the
fields needed to render the in-app reader (hero image, title, trust-score
dial, summary, source link). Subsequent slices extend ``ItemDetailView``
with genealogy (ARG-159), pgvector similarity (ARG-160), and
track_history (ARG-161) without changing the T1 contract.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem


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

    return ItemDetailView(
        id=row.id,
        title=row.title,
        source_url=row.source_url,
        image_url=row.image_url,
        summary=row.summary,
        category=row.category,
        trust_score=row.trust_score,
        published_at=row.published_at,
    )
