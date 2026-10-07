"""사건 경계 교정 반영 — ARG-245.

재군집(`recluster.recluster_period`)이 낸 합칠·가를 후보를 실제 사건 층에 쓴다.
읽기 전용인 후보 계산과 달리 여기는 쓰기이므로 세 가지를 못 박는다:

- **사건을 지우지 않는다.** 흡수되는 사건은 `merged_into_id`만 채운 툼스톤으로
  남는다 — 옛 링크가 사는 유일한 방법이다(`TechEvent` docstring 참고).
- **한 트랜잭션.** 청소 → 병합 → 분할이 하나의 `session.begin()` 안에서 돈다.
  어디서든 예외가 나면 전부 롤백되고 예외는 호출자에게 그대로 올라간다. 재시도용
  상태는 저장하지 않는다 — 다음 실행이 후보를 새로 계산해 처음부터 다시 한다.
- **매 실행은 청소로 시작한다.** 툼스톤 사건에 아직 붙어 있는 문서·엔티티 링크를
  최종 생존자로 옮긴다. 후보 재확인 직후 낮 배정이 막 툼스톤이 된 사건에 문서를
  붙여도 다음 실행에서 저절로 복구된다.

분할 후보이면서 병합 그룹에도 있는 사건은 그 사건과 병합 그룹 전체를 그날 건너뛴다.
이웃 상한(capped)에 걸린 분할 후보는 반영하지 않고 보고만 한다.

툼스톤 해석은 `services.event_resolution`의 규약(8단계·순환 감지)을 그대로 쓴다.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Union

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from argos.brain.recluster_candidates import (
    MergeCandidate,
    ReclusterCandidates,
    SplitCandidate,
)
from argos.models.entity import EventEntity
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import TechItem
from argos.services.event_resolution import resolve_events

logger = logging.getLogger(__name__)

SKIP_UNRESOLVABLE = "unresolvable_chain"
"""툼스톤 사슬이 순환하거나 8단계를 넘어 살아 있는 사건에 닿지 못했다."""
SKIP_MISSING = "stale_missing"
"""후보가 가리키는 사건이 DB에 없다."""
SKIP_TOMBSTONED = "stale_tombstoned"
"""후보 계산 뒤 사건이 이미 툼스톤이 됐다."""
SKIP_DOCUMENTS_CHANGED = "stale_documents_changed"
"""후보의 근거 문서가 더 이상 그 사건에 붙어 있지 않다."""

SKIP_CAPPED = "capped"
"""이웃 상한에 걸린 분할 후보(ARG-283) — 내용이 아니라 상한 탓일 수 있어 보고만 한다."""
SKIP_CONFLICT = "split_merge_conflict"
"""같은 밤 분할 후보이면서 병합 그룹에도 있다 — 병합 쌍은 분할 전 사건 기준이라
먼저 가르면 어느 조각을 합칠지 정해지지 않는다. 다음 밤에 새로 판단한다."""

LinkModel = Union[type[EventDocument], type[EventEntity]]


@dataclass(frozen=True)
class LinkRelocation:
    """사건 링크 한 줄을 옮긴 기록."""

    kind: Literal["document", "entity"]
    target_id: uuid.UUID
    from_event_id: uuid.UUID
    to_event_id: uuid.UUID
    deduplicated: bool
    """참이면 도착 사건에 이미 같은 링크가 있어 출발 쪽 행을 지웠다."""


@dataclass(frozen=True)
class SkippedCandidate:
    """반영하지 않은 대상과 그 사유. 사람이 다음 날 확인할 수 있게 남긴다."""

    kind: Literal["cleanup", "merge", "split"]
    event_ids: tuple[uuid.UUID, ...]
    reason: str


@dataclass(frozen=True)
class MergeGroup:
    """이어진 병합 쌍을 묶은 한 그룹. A–B, B–C → (A, B, C)."""

    event_ids: tuple[uuid.UUID, ...]
    """오름차순 정렬된 그룹 구성 사건."""
    candidates: tuple[MergeCandidate, ...]
    """그룹을 이룬 후보 쌍들, 입력 순서."""


@dataclass(frozen=True)
class AppliedMerge:
    """반영된 병합 한 건."""

    survivor_id: uuid.UUID
    absorbed_ids: tuple[uuid.UUID, ...]
    document_count: int
    """병합 후 생존자에 붙은 서로 다른 근거 문서 수(= 출처 개수)."""


@dataclass(frozen=True)
class AppliedSplit:
    event_id: uuid.UUID
    """원래 사건 id — 가장 큰 조각이 그대로 갖는다."""
    new_event_ids: tuple[uuid.UUID, ...]


@dataclass
class CorrectionResult:
    """한 번의 교정 실행이 바꾼 것과 건너뛴 것."""

    relocations: list[LinkRelocation] = field(default_factory=list)
    skipped: list[SkippedCandidate] = field(default_factory=list)
    merges: list[AppliedMerge] = field(default_factory=list)
    splits: list[AppliedSplit] = field(default_factory=list)


def _target_column(model: LinkModel):
    return model.tech_item_id if model is EventDocument else model.entity_id


def _kind(model: LinkModel) -> Literal["document", "entity"]:
    return "document" if model is EventDocument else "entity"


async def _move_links(
    session: AsyncSession,
    model: LinkModel,
    from_event_id: uuid.UUID,
    to_event_id: uuid.UUID,
) -> list[tuple[uuid.UUID, bool]]:
    """`from` 사건의 링크를 전부 `to` 사건으로 옮긴다.

    도착 사건에 이미 있는 대상은 유니크 제약을 깨지 않도록 출발 쪽 행을 지운다.
    Returns: `(대상 id, 중복이라 지웠는가)` 목록, 대상 id 오름차순.
    """
    target = _target_column(model)
    targets = set(
        (await session.execute(select(target).where(model.event_id == from_event_id))).scalars()
    )
    if not targets:
        return []
    existing = set(
        (
            await session.execute(
                select(target).where(model.event_id == to_event_id, target.in_(targets))
            )
        ).scalars()
    )
    if existing:
        await session.execute(
            delete(model)
            .where(model.event_id == from_event_id, target.in_(existing))
            .execution_options(synchronize_session=False)
        )
    movable = targets - existing
    if movable:
        await session.execute(
            update(model)
            .where(model.event_id == from_event_id, target.in_(movable))
            .values(event_id=to_event_id)
            .execution_options(synchronize_session=False)
        )
    return [(target_id, target_id in existing) for target_id in sorted(targets)]


async def _relocate(
    session: AsyncSession,
    result: CorrectionResult,
    from_event_id: uuid.UUID,
    to_event_id: uuid.UUID,
) -> bool:
    """문서·엔티티 링크를 옮기고 기록한다. 문서가 하나라도 옮겨졌으면 참."""
    moved_documents = False
    for model in (EventDocument, EventEntity):
        for target_id, deduplicated in await _move_links(
            session, model, from_event_id, to_event_id
        ):
            result.relocations.append(
                LinkRelocation(
                    kind=_kind(model),
                    target_id=target_id,
                    from_event_id=from_event_id,
                    to_event_id=to_event_id,
                    deduplicated=deduplicated,
                )
            )
            if model is EventDocument:
                moved_documents = True
    return moved_documents


async def _mark_naming_stale(session: AsyncSession, event_id: uuid.UUID) -> None:
    await session.execute(
        update(TechEvent)
        .where(TechEvent.id == event_id)
        .values(naming_stale=True)
        .execution_options(synchronize_session=False)
    )


async def _cleanup_tombstone_links(session: AsyncSession, result: CorrectionResult) -> None:
    """툼스톤 사건에 남은 링크를 최종 생존자로 옮긴다."""
    tombstoned_with_links: set[uuid.UUID] = set()
    for model in (EventDocument, EventEntity):
        tombstoned_with_links |= set(
            (
                await session.execute(
                    select(model.event_id)
                    .join(TechEvent, TechEvent.id == model.event_id)
                    .where(TechEvent.merged_into_id.is_not(None))
                    .distinct()
                )
            ).scalars()
        )
    if not tombstoned_with_links:
        return

    finals = await resolve_events(session, tombstoned_with_links)
    still_tombstoned = set(
        (
            await session.execute(
                select(TechEvent.id).where(
                    TechEvent.id.in_(set(finals.values())),
                    TechEvent.merged_into_id.is_not(None),
                )
            )
        ).scalars()
    )

    for event_id in sorted(tombstoned_with_links):
        final_id = finals[event_id]
        if final_id in still_tombstoned or final_id == event_id:
            logger.warning(
                "Tombstone %s never reaches a live event (stopped at %s); leaving its links.",
                event_id,
                final_id,
            )
            result.skipped.append(
                SkippedCandidate(kind="cleanup", event_ids=(event_id,), reason=SKIP_UNRESOLVABLE)
            )
            continue
        if await _relocate(session, result, event_id, final_id):
            await _mark_naming_stale(session, final_id)


def group_merge_candidates(merges: Sequence[MergeCandidate]) -> list[MergeGroup]:
    """병합 쌍을 연결 성분으로 묶는다. 결과는 그룹의 가장 작은 사건 id 순."""
    parent: dict[uuid.UUID, uuid.UUID] = {}

    def find(x: uuid.UUID) -> uuid.UUID:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for candidate in merges:
        left, right = candidate.event_ids
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[max(root_l, root_r)] = min(root_l, root_r)

    members: dict[uuid.UUID, set[uuid.UUID]] = {}
    for event_id in list(parent):
        members.setdefault(find(event_id), set()).add(event_id)
    grouped: dict[uuid.UUID, list[MergeCandidate]] = {}
    for candidate in merges:
        grouped.setdefault(find(candidate.event_ids[0]), []).append(candidate)

    return sorted(
        (
            MergeGroup(event_ids=tuple(sorted(ids)), candidates=tuple(grouped[root]))
            for root, ids in members.items()
        ),
        key=lambda group: group.event_ids[0],
    )


def choose_survivor(stats: Mapping[uuid.UUID, tuple[int, datetime]]) -> uuid.UUID:
    """생존자 선택: 문서 수 많은 쪽 → 이른 occurred_at → 작은 id."""
    return min(stats, key=lambda event_id: (-stats[event_id][0], stats[event_id][1], event_id))


async def _apply_merge_group(
    session: AsyncSession, group: MergeGroup, result: CorrectionResult
) -> None:
    """한 병합 그룹을 재확인한 뒤 반영한다. 낡았으면 사유와 함께 건너뛴다."""
    rows = (
        await session.execute(
            select(TechEvent.id, TechEvent.merged_into_id, TechEvent.occurred_at).where(
                TechEvent.id.in_(group.event_ids)
            )
        )
    ).all()
    if len(rows) != len(group.event_ids):
        result.skipped.append(SkippedCandidate("merge", group.event_ids, SKIP_MISSING))
        return
    if any(row.merged_into_id is not None for row in rows):
        result.skipped.append(SkippedCandidate("merge", group.event_ids, SKIP_TOMBSTONED))
        return
    for candidate in group.candidates:
        evidence = set(candidate.evidence_document_ids)
        if not evidence:
            continue
        linked = set(
            (
                await session.execute(
                    select(EventDocument.tech_item_id).where(
                        EventDocument.event_id.in_(candidate.event_ids),
                        EventDocument.tech_item_id.in_(evidence),
                    )
                )
            ).scalars()
        )
        if linked != evidence:
            result.skipped.append(
                SkippedCandidate("merge", group.event_ids, SKIP_DOCUMENTS_CHANGED)
            )
            return

    counts = dict(
        (
            await session.execute(
                select(EventDocument.event_id, func.count())
                .where(EventDocument.event_id.in_(group.event_ids))
                .group_by(EventDocument.event_id)
            )
        ).all()
    )
    stats = {row.id: (counts.get(row.id, 0), row.occurred_at) for row in rows}
    survivor_id = choose_survivor(stats)
    absorbed = tuple(event_id for event_id in group.event_ids if event_id != survivor_id)

    for absorbed_id in absorbed:
        await _relocate(session, result, absorbed_id, survivor_id)
        # 사슬 압축: 흡수 사건을 가리키던 기존 툼스톤도 생존자를 직접 가리킨다.
        await session.execute(
            update(TechEvent)
            .where(TechEvent.merged_into_id == absorbed_id)
            .values(merged_into_id=survivor_id)
            .execution_options(synchronize_session=False)
        )
        await session.execute(
            update(TechEvent)
            .where(TechEvent.id == absorbed_id)
            .values(merged_into_id=survivor_id)
            .execution_options(synchronize_session=False)
        )

    await session.execute(
        update(TechEvent)
        .where(TechEvent.id == survivor_id)
        .values(occurred_at=min(row.occurred_at for row in rows))
        .execution_options(synchronize_session=False)
    )
    await _mark_naming_stale(session, survivor_id)
    document_count = await session.scalar(
        select(func.count(func.distinct(EventDocument.tech_item_id))).where(
            EventDocument.event_id == survivor_id
        )
    )
    result.merges.append(
        AppliedMerge(
            survivor_id=survivor_id,
            absorbed_ids=absorbed,
            document_count=document_count or 0,
        )
    )


def choose_largest_fragment(
    fragments: Sequence[Sequence[uuid.UUID]], document_times: Mapping[uuid.UUID, datetime]
) -> int:
    """기간 안 문서 수 많은 조각 → 이른 시각 → 작은 문서 id."""
    return min(
        range(len(fragments)),
        key=lambda i: (
            -len(fragments[i]),
            min(document_times[d] for d in fragments[i]),
            min(fragments[i]),
        ),
    )


def find_conflicts(
    groups: Sequence[MergeGroup], splits: Sequence[SplitCandidate]
) -> tuple[list[MergeGroup], list[MergeGroup], frozenset[uuid.UUID]]:
    split_ids = {split.event_id for split in splits}
    clean: list[MergeGroup] = []
    conflicted: list[MergeGroup] = []
    blocked: set[uuid.UUID] = set()
    for group in groups:
        overlap = split_ids.intersection(group.event_ids)
        if overlap:
            conflicted.append(group)
            blocked |= overlap
        else:
            clean.append(group)
    return clean, conflicted, frozenset(blocked)


async def _apply_split(
    session: AsyncSession, split: SplitCandidate, result: CorrectionResult
) -> None:
    key = (split.event_id,)
    row = (
        await session.execute(
            select(TechEvent.id, TechEvent.merged_into_id).where(TechEvent.id == split.event_id)
        )
    ).first()
    if row is None:
        result.skipped.append(SkippedCandidate("split", key, SKIP_MISSING))
        return
    if row.merged_into_id is not None:
        result.skipped.append(SkippedCandidate("split", key, SKIP_TOMBSTONED))
        return
    fragments = [tuple(group) for group in split.groups if group]
    all_documents = {d for fragment in fragments for d in fragment}
    linked = set(
        (
            await session.execute(
                select(EventDocument.tech_item_id).where(
                    EventDocument.event_id == split.event_id,
                    EventDocument.tech_item_id.in_(all_documents),
                )
            )
        ).scalars()
    )
    if len(fragments) < 2 or linked != all_documents:
        result.skipped.append(SkippedCandidate("split", key, SKIP_DOCUMENTS_CHANGED))
        return

    document_times = dict(
        (
            await session.execute(
                select(TechItem.id, func.coalesce(TechItem.published_at, TechItem.created_at)).where(
                    TechItem.id.in_(all_documents)
                )
            )
        ).all()
    )
    keep = choose_largest_fragment(fragments, document_times)
    new_ids: list[uuid.UUID] = []
    for index, fragment in enumerate(fragments):
        if index == keep:
            continue
        new_event = TechEvent(
            title=None,
            summary=None,
            occurred_at=min(document_times[d] for d in fragment),
            naming_stale=True,
        )
        session.add(new_event)
        await session.flush()
        await session.execute(
            update(EventDocument)
            .where(
                EventDocument.event_id == split.event_id,
                EventDocument.tech_item_id.in_(fragment),
            )
            .values(event_id=new_event.id)
            .execution_options(synchronize_session=False)
        )
        new_ids.append(new_event.id)
    await _mark_naming_stale(session, split.event_id)
    result.splits.append(AppliedSplit(event_id=split.event_id, new_event_ids=tuple(new_ids)))


async def apply_corrections(
    session_factory: async_sessionmaker[AsyncSession],
    candidates: ReclusterCandidates | None = None,
) -> CorrectionResult:
    """청소 → 병합 → 분할을 한 트랜잭션으로 반영한다.

    예외가 나면 전부 롤백하고 예외를 다시 던진다 — 호출자(ARG-246 스케줄러)가
    잡아서 평소 동작을 막지 않게 한다.
    """
    if candidates is None:
        candidates = ReclusterCandidates(merges=(), splits=())
    result = CorrectionResult()
    async with session_factory() as session:
        async with session.begin():
            await _cleanup_tombstone_links(session, result)
            groups = group_merge_candidates(candidates.merges)
            clean_groups, conflicted_groups, blocked = find_conflicts(groups, candidates.splits)
            for group in conflicted_groups:
                result.skipped.append(SkippedCandidate("merge", group.event_ids, SKIP_CONFLICT))
            for group in clean_groups:
                await _apply_merge_group(session, group, result)
            for split in sorted(candidates.splits, key=lambda s: s.event_id):
                if split.capped:
                    result.skipped.append(SkippedCandidate("split", (split.event_id,), SKIP_CAPPED))
                elif split.event_id in blocked:
                    result.skipped.append(SkippedCandidate("split", (split.event_id,), SKIP_CONFLICT))
                else:
                    await _apply_split(session, split, result)
    return result
