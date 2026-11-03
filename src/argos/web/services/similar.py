"""'비슷한 역할' 읽기 서비스 (ARG-327) — 기술 페이지의 같은 아래 갈래 기술들.

같은 아래 갈래로 배정된 다른 기술 중 판정이 ``Technology``인 대표 이름만
후보다. 같은 이름의 변형(포함 관계)은 '비슷한 기술'이 아니라 같은 기술이므로
뺀다. 근거 '기사에서 함께 나열됨'은 배정 때 저장한 ``evidence["neighbors"]``를
읽기만 한다 — 읽기 전용이고 LLM·마이그레이션 없음.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

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
from argos.web.services.technology import _UNTITLED_EVENT, BranchRef, TechEventRef, TechRef

SIMILAR_LIMIT = 5


@dataclass(frozen=True)
class SimilarTech:
    entity_id: uuid.UUID
    name: str
    branch: BranchRef
    latest_event: Optional[TechEventRef]
    # 0 = '함께 나열됨' 근거 없음.
    colisted_count: int = 0
    colisted_example: Optional[str] = None

    @property
    def href(self) -> str:
        return f"/tech/{self.entity_id}"


def _neighbors(evidence: Any) -> list[tuple[str, int, Optional[str]]]:
    """``evidence["neighbors"]``에서 (이름 키, 횟수, 첫 예문)만 꺼낸다. 모양이 다르면 건너뛴다."""
    if not isinstance(evidence, dict):
        return []
    raw = evidence.get("neighbors")
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, int, Optional[str]]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name, count = item.get("name"), item.get("count")
        if not isinstance(name, str) or isinstance(count, bool) or not isinstance(count, int):
            continue
        if count <= 0:
            continue
        examples = item.get("examples")
        example = None
        if isinstance(examples, list) and examples and isinstance(examples[0], str):
            example = examples[0] or None
        out.append((name, count, example))
    return out


def _contained(a: str, b: str) -> bool:
    """낱말 단위로 연속 포함(양방향)이거나 같으면 True."""
    return f" {a} " in f" {b} " or f" {b} " in f" {a} "


def _colisted(
    base_keys: set[str],
    base_evidence: Any,
    cand_keys: set[str],
    cand_evidence: Any,
) -> tuple[int, Optional[str]]:
    """나열은 대칭이라 어느 쪽 행에 있어도 인정한다. 일치 항목 count의 최댓값과 그 예문."""
    best_count, best_example = 0, None
    for name, count, example in _neighbors(base_evidence):
        if name in cand_keys and count > best_count:
            best_count, best_example = count, example
    for name, count, example in _neighbors(cand_evidence):
        if name in base_keys and count > best_count:
            best_count, best_example = count, example
    return best_count, best_example


async def _key_bundles(
    session: AsyncSession, canonical_ids: list[uuid.UUID]
) -> dict[uuid.UUID, set[str]]:
    """대표별 이름 묶음 = 대표 키 + 그 별칭들의 키."""
    bundles: dict[uuid.UUID, set[str]] = {}
    for cid, key in (
        await session.execute(
            select(Entity.id, Entity.normalized_key).where(Entity.id.in_(canonical_ids))
        )
    ).all():
        bundles.setdefault(cid, set()).add(key)
    for cid, key in (
        await session.execute(
            select(EntityAlias.canonical_entity_id, Entity.normalized_key)
            .join(Entity, Entity.id == EntityAlias.alias_entity_id)
            .where(EntityAlias.canonical_entity_id.in_(canonical_ids))
        )
    ).all():
        bundles.setdefault(cid, set()).add(key)
    return bundles


async def _latest_events(
    session: AsyncSession, canonical_ids: list[uuid.UUID]
) -> dict[uuid.UUID, TechEventRef]:
    """대표별 가장 최근 사건 — 한 쿼리로 모두 채운다. 별칭 언급은 대표로 접고 툼스톤은 뺀다."""
    if not canonical_ids:
        return {}
    canon_id = func.coalesce(EntityAlias.canonical_entity_id, DocumentEntity.entity_id)
    rows = (
        await session.execute(
            select(
                canon_id.label("canon_id"),
                TechEvent.id,
                TechEvent.title,
                TechEvent.occurred_at,
                TechEvent.event_type,
            )
            .select_from(TechEvent)
            .join(EventDocument, EventDocument.event_id == TechEvent.id)
            .join(DocumentEntity, DocumentEntity.tech_item_id == EventDocument.tech_item_id)
            .outerjoin(EntityAlias, EntityAlias.alias_entity_id == DocumentEntity.entity_id)
            .where(canon_id.in_(canonical_ids), TechEvent.merged_into_id.is_(None))
            .distinct(canon_id)
            .order_by(canon_id, TechEvent.occurred_at.desc(), TechEvent.id.desc())
        )
    ).all()
    return {
        cid: TechEventRef(
            id=eid,
            title=title or _UNTITLED_EVENT,
            occurred_at=occurred_at,
            event_type_label=type_label(event_type),
        )
        for cid, eid, title, occurred_at, event_type in rows
    }


class _Lookup:
    """한 번의 조회 안에서 갈래 단위 데이터를 공유하는 캐시.

    같은 아래 갈래의 칩들은 후보·이름 묶음·최근 사건을 한 번만 읽는다 — 쿼리
    수가 칩 수가 아니라 서로 다른 갈래 수에 따라 늘도록.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.assignments: dict[uuid.UUID, Optional[RoleAssignment]] = {}
        self.branches: dict[uuid.UUID, Optional[RoleBranch]] = {}
        self.candidates: dict[uuid.UUID, list[tuple[uuid.UUID, str, Any]]] = {}
        self.bundles: dict[uuid.UUID, set[str]] = {}
        self.events: dict[uuid.UUID, TechEventRef] = {}
        self._events_done: set[uuid.UUID] = set()

    async def load_assignments(self, ids: Sequence[uuid.UUID]) -> None:
        missing = [i for i in dict.fromkeys(ids) if i not in self.assignments]
        if not missing:
            return
        found = {
            ra.entity_id: ra
            for ra in (
                await self.session.execute(
                    select(RoleAssignment).where(RoleAssignment.entity_id.in_(missing))
                )
            )
            .scalars()
            .all()
        }
        for i in missing:
            self.assignments[i] = found.get(i)
        need = {
            ra.branch_id
            for ra in found.values()
            if ra.branch_id is not None and ra.branch_id not in self.branches
        }
        if need:
            got = {
                br.id: br
                for br in (
                    await self.session.execute(select(RoleBranch).where(RoleBranch.id.in_(need)))
                )
                .scalars()
                .all()
            }
            for bid in need:
                self.branches[bid] = got.get(bid)

    async def load_candidates(self, branch_id: uuid.UUID) -> list[tuple[uuid.UUID, str, Any]]:
        if branch_id not in self.candidates:
            rows = (
                await self.session.execute(
                    select(Entity.id, Entity.name, RoleAssignment.evidence)
                    .join(RoleAssignment, RoleAssignment.entity_id == Entity.id)
                    .join(TechnologyProfile, TechnologyProfile.entity_id == Entity.id)
                    .where(
                        RoleAssignment.branch_id == branch_id,
                        TechnologyProfile.verdict == TechVerdict.TECHNOLOGY,
                        Entity.id.not_in(select(EntityAlias.alias_entity_id)),
                    )
                )
            ).all()
            self.candidates[branch_id] = [(r[0], r[1], r[2]) for r in rows]
        return self.candidates[branch_id]

    async def load_bundles(self, ids: Sequence[uuid.UUID]) -> None:
        missing = [i for i in dict.fromkeys(ids) if i not in self.bundles]
        if not missing:
            return
        got = await _key_bundles(self.session, missing)
        for i in missing:
            self.bundles[i] = got.get(i, set())

    async def load_events(self, ids: Sequence[uuid.UUID]) -> None:
        missing = [i for i in dict.fromkeys(ids) if i not in self._events_done]
        if not missing:
            return
        self.events.update(await _latest_events(self.session, missing))
        self._events_done.update(missing)


async def _similar(lookup: _Lookup, canonical_id: uuid.UUID) -> tuple[SimilarTech, ...]:
    await lookup.load_assignments([canonical_id])
    base = lookup.assignments.get(canonical_id)
    if base is None or base.branch_id is None:
        return ()
    branch = lookup.branches.get(base.branch_id)
    if branch is None or branch.parent_id is None:
        return ()

    rows = [r for r in await lookup.load_candidates(base.branch_id) if r[0] != canonical_id]
    if not rows:
        return ()

    await lookup.load_bundles([canonical_id] + [r[0] for r in rows])
    base_keys = lookup.bundles.get(canonical_id, set())

    kept = []
    for entity_id, name, evidence in rows:
        cand_keys = lookup.bundles.get(entity_id, set())
        if any(_contained(a, b) for a in base_keys for b in cand_keys):
            continue
        count, example = _colisted(base_keys, base.evidence, cand_keys, evidence)
        kept.append((entity_id, name, count, example))

    await lookup.load_events([k[0] for k in kept])
    events = lookup.events
    kept.sort(
        key=lambda k: (
            0 if k[2] > 0 else 1,
            1 if k[0] not in events else 0,
            -events[k[0]].occurred_at.timestamp() if k[0] in events else 0.0,
            k[1].casefold(),
            str(k[0]),
        )
    )
    ref = BranchRef(id=branch.id, name=branch.name)
    return tuple(
        SimilarTech(
            entity_id=entity_id,
            name=name,
            branch=ref,
            latest_event=events.get(entity_id),
            colisted_count=count,
            colisted_example=example,
        )
        for entity_id, name, count, example in kept[:SIMILAR_LIMIT]
    )


async def fetch_similar_techs(
    session: AsyncSession, canonical_id: uuid.UUID
) -> tuple[SimilarTech, ...]:
    """같은 아래 갈래의 다른 기술 최대 ``SIMILAR_LIMIT``개.

    미분류이거나 위 갈래에 배정된 기술은 '같은 역할'이 너무 넓어 ``()``.
    정렬: 함께 나열된 근거가 있는 것 먼저 -> 최근 사건 새로운 순(없으면 뒤) -> 이름.
    """
    return await _similar(_Lookup(session), canonical_id)


@dataclass(frozen=True)
class MixedEventRef:
    """섞어 보기 한 줄 — 사건 하나와 그 사건에 언급된 기술들."""

    id: uuid.UUID
    title: str
    occurred_at: datetime
    event_type_label: Optional[str]
    techs: tuple[TechRef, ...]  # 이 기술 먼저, 그다음 similar 순서


async def fetch_mixed_timeline(
    session: AsyncSession, own: TechRef, similar: Sequence[SimilarTech]
) -> tuple[MixedEventRef, ...]:
    """이 기술과 비슷한 기술들의 사건을 시간순(최신 먼저)으로 한 목록에 섞는다.

    한 쿼리로 가져온다. 별칭 언급은 대표로 접고 툼스톤은 뺀다. 두 기술이
    같은 사건에 나오면 한 줄에 이름을 모아 둔다.
    """
    order = [own, *(TechRef(entity_id=s.entity_id, name=s.name) for s in similar)]
    refs = {t.entity_id: t for t in order}
    rank = {t.entity_id: i for i, t in enumerate(order)}
    canon_id = func.coalesce(EntityAlias.canonical_entity_id, DocumentEntity.entity_id)
    rows = (
        await session.execute(
            select(
                canon_id.label("canon_id"),
                TechEvent.id,
                TechEvent.title,
                TechEvent.occurred_at,
                TechEvent.event_type,
            )
            .select_from(TechEvent)
            .join(EventDocument, EventDocument.event_id == TechEvent.id)
            .join(DocumentEntity, DocumentEntity.tech_item_id == EventDocument.tech_item_id)
            .outerjoin(EntityAlias, EntityAlias.alias_entity_id == DocumentEntity.entity_id)
            .where(canon_id.in_(list(refs)), TechEvent.merged_into_id.is_(None))
            .distinct()
            .order_by(TechEvent.occurred_at.desc(), TechEvent.id.desc())
        )
    ).all()
    grouped: dict[uuid.UUID, tuple] = {}
    for cid, eid, title, occurred_at, event_type in rows:
        if eid not in grouped:
            grouped[eid] = (title, occurred_at, event_type, set())
        grouped[eid][3].add(cid)
    return tuple(
        MixedEventRef(
            id=eid,
            title=title or _UNTITLED_EVENT,
            occurred_at=occurred_at,
            event_type_label=type_label(event_type),
            techs=tuple(refs[c] for c in sorted(cids, key=rank.__getitem__)),
        )
        for eid, (title, occurred_at, event_type, cids) in grouped.items()
    )


SIMILAR_NEWS_LIMIT = 3


@dataclass(frozen=True)
class SimilarNews:
    """사건 상세 '비슷한 기술의 최근 소식' 한 줄 — 비슷한 기술 하나와 그 최근 사건."""

    entity_id: uuid.UUID
    name: str
    branch: BranchRef
    event: TechEventRef

    @property
    def href(self) -> str:
        return f"/tech/{self.entity_id}"


async def fetch_event_similar_news(
    session: AsyncSession, event_id: uuid.UUID, chip_entity_ids: Sequence[uuid.UUID]
) -> tuple[SimilarNews, ...]:
    """사건에 등장한 기술들의 '비슷한 기술'이 최근에 낸 사건 최대 ``SIMILAR_NEWS_LIMIT``건.

    칩 기술 자체와 이 사건, 최근 사건이 없는 기술은 뺀다. 같은 비슷한 기술·같은
    사건은 한 번만. 최신순(``occurred_at DESC, id DESC``).
    """
    if not chip_entity_ids:
        return ()
    chip_set = set(chip_entity_ids)
    seen_techs: set[uuid.UUID] = set()
    seen_events: set[uuid.UUID] = {event_id}
    found: list[SimilarNews] = []
    lookup = _Lookup(session)
    await lookup.load_assignments(chip_entity_ids)
    await lookup.load_bundles(chip_entity_ids)
    for chip_id in chip_entity_ids:
        for tech in await _similar(lookup, chip_id):
            if tech.entity_id in chip_set or tech.entity_id in seen_techs:
                continue
            ev = tech.latest_event
            if ev is None or ev.id in seen_events:
                continue
            seen_techs.add(tech.entity_id)
            seen_events.add(ev.id)
            found.append(
                SimilarNews(entity_id=tech.entity_id, name=tech.name, branch=tech.branch, event=ev)
            )
    found.sort(key=lambda n: (n.event.occurred_at, str(n.event.id)), reverse=True)
    return tuple(found[:SIMILAR_NEWS_LIMIT])
