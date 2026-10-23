"""확정된 역할 갈래의 저장 — ARG-311.

``role_branches`` 테이블이 확정본이다. 이 모듈의 **쓰기 경로는 ``apply_plan`` 하나**
이고, 그것을 부르는 곳은 ``argos.taxonomy_cli``(사람이 친 ``argos taxonomy apply``)
뿐이다. 자동 처리(크롤·교정·스케줄)는 ``load_confirmed``만 쓴다 — 이 불변식은
``tests/brain/test_taxonomy_isolation.py``가 지킨다.

``plan_apply``는 현재 확정본과 새 트리를 **노드 id 기준**으로 비교하는 순수 함수다.
이름만 바뀐 노드는 renamed(삭제+추가가 아님)라서 id가 유지되고, 그래서 ARG-313의
배정 행이 깨지지 않는다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.taxonomy import Branch, TaxonomyError, Tree, validate
from argos.models.role_branch import RoleBranch


@dataclass(frozen=True)
class ApplyPlan:
    """현재 확정본 -> 새 트리의 차이. 비어 있으면 쓸 것이 없다."""

    target: Tree
    added: tuple[tuple[str, str | None], ...] = ()  # (이름, 부모 이름 | None)
    renamed: tuple[tuple[str, str], ...] = ()  # (옛 이름, 새 이름)
    redescribed: tuple[str, ...] = ()  # 설명이 바뀐 갈래(새 이름)
    moved: tuple[tuple[str, str | None, str | None], ...] = ()  # (이름, 옛 부모, 새 부모)
    reordered: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    removed_ids: frozenset[str] = field(default_factory=frozenset)
    current_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not (
            self.added
            or self.renamed
            or self.redescribed
            or self.moved
            or self.reordered
            or self.removed
        )

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for name, parent in self.added:
            where = f"{parent!r} 아래" if parent else "위 갈래"
            lines.append(f"  + 추가: {name!r} ({where})")
        for old, new in self.renamed:
            lines.append(f"  ~ 이름 변경: {old!r} -> {new!r}")
        for name in self.redescribed:
            lines.append(f"  ~ 설명 변경: {name!r}")
        for name, old_p, new_p in self.moved:
            lines.append(f"  > 이동: {name!r} ({old_p or '위 갈래'} -> {new_p or '위 갈래'})")
        for name in self.reordered:
            lines.append(f"  ~ 순서 변경: {name!r}")
        for name in self.removed:
            lines.append(f"  - 삭제: {name!r}")
        return lines


def _flatten(tree: Tree) -> dict[str, tuple[Branch, str | None, int]]:
    """id가 있는 노드: id -> (노드, 부모 id, position)."""
    out: dict[str, tuple[Branch, str | None, int]] = {}
    for i, top in enumerate(tree):
        if top.id:
            out[top.id] = (top, None, i)
        for j, child in enumerate(top.children):
            if child.id:
                out[child.id] = (child, top.id, j)
    return out


def plan_apply(current: Tree | None, new: Tree) -> ApplyPlan:
    """현재 확정본(없으면 None)과 새 트리를 비교한다. 순수 함수."""
    errors = validate(new)
    if errors:
        raise TaxonomyError(errors)

    cur = _flatten(current or ())
    cur_names = {bid: node.name for bid, (node, _p, _i) in cur.items()}
    new_flat = _flatten(new)
    if current is not None:
        stale = sorted(bid for bid in new_flat if bid not in cur)
        if stale:
            raise TaxonomyError(
                [
                    "파일의 갈래 id가 현재 확정본에 없습니다 — "
                    "`argos taxonomy export`로 다시 받아 편집하세요 "
                    f"(모르는 id {len(stale)}개)"
                ]
            )

    added: list[tuple[str, str | None]] = []
    renamed: list[tuple[str, str]] = []
    redescribed: list[str] = []
    moved: list[tuple[str, str | None, str | None]] = []
    reordered: list[str] = []

    def visit(node: Branch, parent: Branch | None, position: int) -> None:
        bid = node.id if (current is not None and node.id in cur) else None
        if bid is None:
            added.append((node.name, parent.name if parent else None))
        else:
            old, old_parent_id, old_pos = cur[bid]
            if old.name != node.name:
                renamed.append((old.name, node.name))
            if old.description != node.description:
                redescribed.append(node.name)
            new_parent_id = parent.id if parent else None
            if old_parent_id != new_parent_id:
                moved.append(
                    (
                        node.name,
                        cur_names.get(old_parent_id) if old_parent_id else None,
                        parent.name if parent else None,
                    )
                )
            if old_pos != position:
                reordered.append(node.name)
        for j, child in enumerate(node.children):
            visit(child, node, j)

    for i, top in enumerate(new):
        visit(top, None, i)

    kept = set(new_flat) & set(cur)
    removed_ids = frozenset(set(cur) - kept)
    removed = tuple(cur[bid][0].name for bid in cur if bid in removed_ids)

    # 위치만 바뀐 노드는 부모가 바뀌면 위치도 따라 바뀌므로 이동과 중복 표시하지 않는다.
    moved_names = {m[0] for m in moved}
    reordered = [n for n in reordered if n not in moved_names]

    return ApplyPlan(
        target=new,
        added=tuple(added),
        renamed=tuple(renamed),
        redescribed=tuple(redescribed),
        moved=tuple(moved),
        reordered=tuple(reordered),
        removed=removed,
        removed_ids=removed_ids,
        current_ids=frozenset(cur),
    )


async def load_confirmed(session: AsyncSession) -> Tree | None:
    """확정본 트리. 한 행도 없으면 None. 읽기 전용 — 자동 처리도 써도 된다."""
    rows = (
        await session.execute(
            select(RoleBranch).order_by(RoleBranch.position, RoleBranch.name)
        )
    ).scalars().all()
    if not rows:
        return None
    children_of: dict[uuid.UUID, list[RoleBranch]] = {}
    tops: list[RoleBranch] = []
    for row in rows:
        if row.parent_id is None:
            tops.append(row)
        else:
            children_of.setdefault(row.parent_id, []).append(row)

    def to_branch(row: RoleBranch, with_children: bool) -> Branch:
        kids = (
            tuple(to_branch(c, False) for c in children_of.get(row.id, []))
            if with_children
            else ()
        )
        return Branch(
            name=row.name,
            description=row.description,
            id=str(row.id),
            children=kids,
        )

    return tuple(to_branch(t, True) for t in tops)


async def apply_plan(session: AsyncSession, plan: ApplyPlan) -> None:
    """계획을 확정본에 쓴다. 커밋은 호출자 몫(한 트랜잭션).

    이름 유일 제약 때문에 순서가 중요하다: 이름이 바뀌거나 지워질 기존 행을 먼저
    임시 이름으로 비켜 놓고(교환 rename·삭제 후 같은 이름 재사용 대비), 새 행을
    넣고, 기존 행을 최종 값으로 갱신하고, 마지막에 지울 행을 삭제한다. 자식의
    부모 이동은 갱신 단계에서 끝나므로 삭제가 자식을 끌고 가지 않는다.
    """
    new_ids: dict[int, uuid.UUID] = {}  # id(node) -> 새 uuid (id 없는 노드)

    def ensure_id(node: Branch) -> uuid.UUID:
        if node.id and node.id in plan.current_ids:
            return uuid.UUID(node.id)
        return new_ids.setdefault(id(node), uuid.uuid4())

    # 1) 이름이 바뀌거나 지워질 기존 행을 임시 이름으로.
    final_names = {
        node.id: node.name
        for top in plan.target
        for node in (top, *top.children)
        if node.id
    }
    current_rows = (
        await session.execute(select(RoleBranch.id, RoleBranch.name))
    ).all()
    for row_id, row_name in current_rows:
        key = str(row_id)
        if key in plan.removed_ids or final_names.get(key, row_name) != row_name:
            await session.execute(
                update(RoleBranch)
                .where(RoleBranch.id == row_id)
                .values(name=f"~tmp-{uuid.uuid4().hex}")
            )

    # 2) 새 행 (위 갈래 먼저, 그다음 아래 갈래 — FK 순서).
    for i, top in enumerate(plan.target):
        if not (top.id and top.id in plan.current_ids):
            await session.execute(
                insert(RoleBranch).values(
                    id=ensure_id(top),
                    parent_id=None,
                    name=top.name,
                    description=top.description,
                    position=i,
                )
            )
    for i, top in enumerate(plan.target):
        top_id = ensure_id(top)
        for j, child in enumerate(top.children):
            if not (child.id and child.id in plan.current_ids):
                await session.execute(
                    insert(RoleBranch).values(
                        id=ensure_id(child),
                        parent_id=top_id,
                        name=child.name,
                        description=child.description,
                        position=j,
                    )
                )

    # 3) 기존 행을 최종 값으로.
    for i, top in enumerate(plan.target):
        top_id = ensure_id(top)
        if top.id and top.id in plan.current_ids:
            await session.execute(
                update(RoleBranch)
                .where(RoleBranch.id == top_id)
                .values(
                    name=top.name,
                    description=top.description,
                    parent_id=None,
                    position=i,
                    confirmed_at=func.now(),
                )
            )
        for j, child in enumerate(top.children):
            if child.id and child.id in plan.current_ids:
                await session.execute(
                    update(RoleBranch)
                    .where(RoleBranch.id == uuid.UUID(child.id))
                    .values(
                        name=child.name,
                        description=child.description,
                        parent_id=top_id,
                        position=j,
                        confirmed_at=func.now(),
                    )
                )

    # 4) 지울 행.
    if plan.removed_ids:
        await session.execute(
            delete(RoleBranch).where(
                RoleBranch.id.in_([uuid.UUID(i) for i in plan.removed_ids])
            )
        )
