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
from typing import Optional, Sequence

from sqlalchemy import func, select
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

# 피드 카드에 붙는 기술 칩 상한. 사건 상세는 자르지 않는다.
CARD_CHIP_LIMIT = 3


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
class TechChip:
    entity_id: uuid.UUID
    name: str
    doc_count: int

    @property
    def href(self) -> str:
        return f"/tech/{self.entity_id}"


@dataclass(frozen=True)
class TechRef:
    entity_id: uuid.UUID
    name: str

    @property
    def href(self) -> str:
        return f"/tech/{self.entity_id}"


@dataclass(frozen=True)
class BranchGroup:
    branch: BranchRef
    techs: tuple[TechRef, ...]


@dataclass(frozen=True)
class BranchPageView:
    branch: BranchRef
    parent: Optional[BranchRef]
    description: str
    # 위 갈래면 아래 갈래별 그룹, 아래 갈래면 자기 자신 하나.
    groups: tuple[BranchGroup, ...]


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


async def fetch_event_tech_chips(
    session: AsyncSession, event_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[TechChip]]:
    """사건별 기술 칩 — 한 번의 쿼리로 모든 사건을 채운다 (N+1 없음).

    언급은 별칭이면 대표로 접고(``entity_aliases``), 대표의 판정이
    ``Technology``인 것만 남긴다. ``doc_count`` = 그 사건에서 대표·별칭 중
    하나라도 언급한 서로 다른 기사 수. 정렬: doc_count 내림차순, 이름 오름차순.
    기술 이름이 없는 사건은 키가 없다.
    """
    ids = list(event_ids)
    if not ids:
        return {}
    canon_id = func.coalesce(EntityAlias.canonical_entity_id, DocumentEntity.entity_id)
    canon = Entity.__table__.alias("canon_entity")
    doc_count = func.count(func.distinct(EventDocument.tech_item_id))
    rows = (
        await session.execute(
            select(
                EventDocument.event_id,
                canon_id.label("entity_id"),
                canon.c.name,
                doc_count.label("doc_count"),
            )
            .join(DocumentEntity, DocumentEntity.tech_item_id == EventDocument.tech_item_id)
            .outerjoin(EntityAlias, EntityAlias.alias_entity_id == DocumentEntity.entity_id)
            .join(canon, canon.c.id == canon_id)
            .join(TechnologyProfile, TechnologyProfile.entity_id == canon_id)
            .where(
                EventDocument.event_id.in_(ids),
                TechnologyProfile.verdict == TechVerdict.TECHNOLOGY,
            )
            .group_by(EventDocument.event_id, canon_id, canon.c.name)
            .order_by(EventDocument.event_id, doc_count.desc(), canon.c.name.asc())
        )
    ).all()
    chips: dict[uuid.UUID, list[TechChip]] = {}
    for event_id, entity_id, name, count in rows:
        chips.setdefault(event_id, []).append(
            TechChip(entity_id=entity_id, name=name, doc_count=count)
        )
    return chips


async def _branch_techs(
    session: AsyncSession, branch_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[TechRef]]:
    """갈래별 기술 — 배정이 그 갈래이고 대표의 판정이 ``Technology``인 것만, 이름순."""
    rows = (
        await session.execute(
            select(RoleAssignment.branch_id, Entity.id, Entity.name)
            .join(Entity, Entity.id == RoleAssignment.entity_id)
            .join(TechnologyProfile, TechnologyProfile.entity_id == Entity.id)
            .where(
                RoleAssignment.branch_id.in_(list(branch_ids)),
                TechnologyProfile.verdict == TechVerdict.TECHNOLOGY,
                Entity.id.not_in(select(EntityAlias.alias_entity_id)),
            )
        )
    ).all()
    out: dict[uuid.UUID, list[TechRef]] = {b: [] for b in branch_ids}
    for branch_id, entity_id, name in rows:
        out[branch_id].append(TechRef(entity_id=entity_id, name=name))
    for techs in out.values():
        techs.sort(key=lambda t: (t.name.casefold(), str(t.entity_id)))
    return out


async def fetch_branch_page(
    session: AsyncSession, branch_id: uuid.UUID
) -> Optional[BranchPageView]:
    """갈래 페이지 — 위 갈래는 아래 갈래별로 묶고, 아래 갈래는 자기 기술만. 없으면 ``None``."""
    branch = await session.get(RoleBranch, branch_id)
    if branch is None:
        return None
    parent = None
    if branch.parent_id is not None:
        p = await session.get(RoleBranch, branch.parent_id)
        if p is not None:
            parent = BranchRef(id=p.id, name=p.name)
    if parent is None and branch.parent_id is None:
        children = (
            await session.execute(
                select(RoleBranch)
                .where(RoleBranch.parent_id == branch.id)
                .order_by(RoleBranch.position, RoleBranch.name)
            )
        ).scalars().all()
        targets = list(children)
    else:
        targets = [branch]
    techs = await _branch_techs(session, [b.id for b in targets])
    return BranchPageView(
        branch=BranchRef(id=branch.id, name=branch.name),
        parent=parent,
        description=branch.description,
        groups=tuple(
            BranchGroup(
                branch=BranchRef(id=b.id, name=b.name), techs=tuple(techs[b.id])
            )
            for b in targets
        ),
    )
