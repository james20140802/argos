"""Read-side service backing the 사건 상세 screen (ARG-243).

An event page lists every evidence document **in report order** — earliest
first — and marks the first one "최초 보도". There is no "which one is the
original" judgement beyond that (decision 4): report order is a fact, origin
is an opinion.

The caller passes an id that has already gone through ``resolve_event()``;
this module matches it literally and never follows tombstones itself.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from argos.web.services.feed import _domain_of


@dataclass(frozen=True)
class EvidenceDoc:
    id: uuid.UUID
    title: str
    source_url: str
    domain: str
    image_url: Optional[str]
    summary: Optional[str]
    reported_at: datetime
    is_first: bool


@dataclass(frozen=True)
class EventDetailView:
    id: uuid.UUID
    title: str
    summary: Optional[str]
    documents: list[EvidenceDoc]
    # The representative (earliest-reported) document — Keep/Pass act on it.
    rep_id: Optional[uuid.UUID]
    category: Optional[CategoryType]
    trust_score: Optional[float]
    image_url: Optional[str]

    @property
    def source_count(self) -> int:
        return len({d.domain for d in self.documents if d.domain}) or len(self.documents)


async def fetch_event_detail(
    session: AsyncSession, event_id: uuid.UUID
) -> Optional[EventDetailView]:
    """The event and its evidence in report order; ``None`` if no such event."""
    event = await session.get(TechEvent, event_id)
    if event is None:
        return None

    reported = func.coalesce(TechItem.published_at, TechItem.created_at)
    rows = (
        await session.execute(
            select(TechItem, reported.label("reported_at"))
            .join(EventDocument, EventDocument.tech_item_id == TechItem.id)
            .where(EventDocument.event_id == event_id)
            .order_by(reported.asc(), TechItem.id.asc())
        )
    ).all()

    documents = [
        EvidenceDoc(
            id=doc.id,
            title=doc.title,
            source_url=doc.source_url,
            domain=_domain_of(doc.source_url),
            image_url=doc.image_url,
            summary=doc.summary,
            reported_at=reported_at,
            is_first=index == 0,
        )
        for index, (doc, reported_at) in enumerate(rows)
    ]
    rep = rows[0][0] if rows else None
    cover = (rep.image_url if rep else None) or next(
        (d.image_url for d in documents if d.image_url), None
    )
    return EventDetailView(
        id=event.id,
        title=event.title or (rep.title if rep else "이름 없는 사건"),
        summary=event.summary or (rep.summary if rep else None),
        documents=documents,
        rep_id=rep.id if rep else None,
        category=rep.category if rep else None,
        trust_score=rep.trust_score if rep else None,
        image_url=cover,
    )
