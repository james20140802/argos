"""Read-side service backing the 기술 페이지 (ARG-318).

A technology page is addressed by its **canonical** entity: a name that has an
``entity_aliases`` row folds to its representative (no chains are stored), and
only a representative whose ``technology_profiles.verdict`` is ``Technology``
gets a page. Everything here is read-only; the description is shown exactly as
stored and no LLM is called.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.entity_alias import EntityAlias
from argos.models.event_document import EventDocument
from argos.models.role_assignment import RoleAssignment
from argos.models.role_branch import RoleBranch
from argos.models.tech_event import TechEvent
from argos.models.technology_profile import TechnologyProfile, TechVerdict
from argos.web.services.event_sections import type_label

_UNTITLED_EVENT = "이름 없는 사건"


@dataclass(frozen=True)
class BranchRef:
    id: uuid.UUID
    name: str


@dataclass(frozen=True)
class TechEventRef:
    id: uuid.UUID
    title: str
    occurred_at: datetime
    event_type_label: Optional[str]


@dataclass(frozen=True)
class TechPageView:
    entity_id: uuid.UUID
    name: str
    description: Optional[str]
    # 위 갈래 -> 아래 갈래 순. ``()`` = 미분류.
    branch_path: tuple[BranchRef, ...]
    aliases: tuple[str, ...]
    events: tuple[TechEventRef, ...]


async def resolve_canonical_id(
    session: AsyncSession, entity_id: uuid.UUID
) -> Optional[uuid.UUID]:
    """No such entity -> ``None``; an alias -> its representative; else itself."""
    exists = await session.scalar(select(Entity.id).where(Entity.id == entity_id))
    if exists is None:
        return None
    canonical = await session.scalar(
        select(EntityAlias.canonical_entity_id).where(
            EntityAlias.alias_entity_id == entity_id
        )
    )
    return canonical if canonical is not None else entity_id


async def is_technology(session: AsyncSession, canonical_id: uuid.UUID) -> bool:
    verdict = await session.scalar(
        select(TechnologyProfile.verdict).where(
            TechnologyProfile.entity_id == canonical_id
        )
    )
    return verdict == TechVerdict.TECHNOLOGY


async def _branch_path(
    session: AsyncSession, canonical_id: uuid.UUID
) -> tuple[BranchRef, ...]:
    branch_id = await session.scalar(
        select(RoleAssignment.branch_id).where(RoleAssignment.entity_id == canonical_id)
    )
    path: list[BranchRef] = []
    seen: set[uuid.UUID] = set()
    while branch_id is not None and branch_id not in seen:
        seen.add(branch_id)
        branch = await session.get(RoleBranch, branch_id)
        if branch is None:
            break
        path.append(BranchRef(id=branch.id, name=branch.name))
        branch_id = branch.parent_id
    path.reverse()
    return tuple(path)


async def fetch_tech_page(
    session: AsyncSession, canonical_id: uuid.UUID
) -> Optional[TechPageView]:
    """The page for a canonical technology; ``None`` if it is not a Technology."""
    profile = (
        await session.execute(
            select(TechnologyProfile).where(TechnologyProfile.entity_id == canonical_id)
        )
    ).scalar_one_or_none()
    if profile is None or profile.verdict != TechVerdict.TECHNOLOGY:
        return None
    entity = await session.get(Entity, canonical_id)
    if entity is None:
        return None

    # Lazy: technology_names -> argos.brain package __init__ -> argos.database,
    # which must stay out of the web app's import graph (tests/web/test_app.py).
    from argos.services.technology_names import names_for

    names = await names_for(session, entity.normalized_key)
    name = names.display if names is not None else entity.name
    aliases = names.aliases if names is not None else ()

    member_ids = [canonical_id]
    member_ids += list(
        (
            await session.execute(
                select(EntityAlias.alias_entity_id).where(
                    EntityAlias.canonical_entity_id == canonical_id
                )
            )
        ).scalars()
    )
    events = (
        await session.execute(
            select(TechEvent)
            .join(EventDocument, EventDocument.event_id == TechEvent.id)
            .join(DocumentEntity, DocumentEntity.tech_item_id == EventDocument.tech_item_id)
            .where(
                DocumentEntity.entity_id.in_(member_ids),
                TechEvent.merged_into_id.is_(None),
            )
            .distinct()
            .order_by(TechEvent.occurred_at.desc(), TechEvent.id.desc())
        )
    ).scalars().all()

    return TechPageView(
        entity_id=canonical_id,
        name=name,
        description=profile.description or None,
        branch_path=await _branch_path(session, canonical_id),
        aliases=tuple(aliases),
        events=tuple(
            TechEventRef(
                id=e.id,
                title=e.title or _UNTITLED_EVENT,
                occurred_at=e.occurred_at,
                event_type_label=type_label(e.event_type),
            )
            for e in events
        ),
    )
