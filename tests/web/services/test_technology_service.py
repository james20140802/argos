"""ARG-318: 기술 페이지 읽기 서비스 — DB 기반(없으면 self-skip)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from argos.config import settings
from argos.web.services.technology import (
    fetch_branch_page,
    fetch_event_tech_chips,
    fetch_tech_page,
    is_technology,
    resolve_canonical_id,
)
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url

pytestmark = [
    pytest.mark.skipif(
        not _db_reachable(_DB_URL),
        reason="pgvector DB not reachable — skipping ARG-318 DB-backed tests",
    ),
    pytest.mark.asyncio,
]

T0 = datetime(2099, 3, 1, tzinfo=timezone.utc)
TOP = "AI 에이전트 및 어시스턴트"
LEAF = "개인·업무 비서형 에이전트"


@asynccontextmanager
async def _seeded(*, description: str | None = "에이전트형 개인 비서", assign: str = "leaf"):
    """대표/별칭/회사 엔티티와 갈래·사건을 심고 ``(Session, ns)``를 준다."""
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.document_entity import DocumentEntity
    from argos.models.entity import Entity
    from argos.models.entity_alias import AliasRule, EntityAlias
    from argos.models.event_document import EventDocument
    from argos.models.role_assignment import RoleAssignment
    from argos.models.role_branch import RoleBranch
    from argos.models.tech_event import TechEvent
    from argos.models.tech_item import CategoryType, TechItem
    from argos.models.technology_profile import TechnologyProfile, TechVerdict

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    sfx = uuid.uuid4().hex[:8]
    ns = type("NS", (), {})()
    entity_ids: list[uuid.UUID] = []
    branch_ids: list[uuid.UUID] = []
    item_ids: list[uuid.UUID] = []
    event_ids: list[uuid.UUID] = []

    try:
        async with Session() as s:
            def ent(key: str, name: str) -> Entity:
                e = Entity(name=name, normalized_key=f"{key} {sfx}")
                s.add(e)
                return e

            canon = ent("claude duet", f"Claude Duet {sfx}")
            alias = ent("duet", f"Duet {sfx}")
            company = ent("anthropic", f"Anthropic {sfx}")
            bare = ent("bare", f"Bare {sfx}")
            await s.flush()
            entity_ids += [canon.id, alias.id, company.id, bare.id]
            s.add(EntityAlias(alias_entity_id=alias.id, canonical_entity_id=canon.id, rule=AliasRule.MANUAL))
            s.add(TechnologyProfile(entity_id=canon.id, verdict=TechVerdict.TECHNOLOGY, description=description))
            s.add(TechnologyProfile(entity_id=company.id, verdict=TechVerdict.NOT_TECHNOLOGY))

            top = RoleBranch(name=f"{TOP} {sfx}", description="위")
            s.add(top)
            await s.flush()
            leaf = RoleBranch(name=f"{LEAF} {sfx}", description="아래", parent_id=top.id)
            s.add(leaf)
            await s.flush()
            branch_ids += [top.id, leaf.id]
            if assign == "leaf":
                s.add(RoleAssignment(entity_id=canon.id, branch_id=leaf.id))
            elif assign == "null":
                s.add(RoleAssignment(entity_id=canon.id, branch_id=None, unclassified_reason="x"))

            async def event(offset_days: int, via: Entity, merged_into=None) -> TechEvent:
                item = TechItem(
                    title=f"doc {uuid.uuid4().hex[:6]}",
                    source_url=f"https://e.example/arg318/{uuid.uuid4()}",
                    raw_content="x",
                    category=CategoryType.MAINSTREAM,
                )
                s.add(item)
                await s.flush()
                item_ids.append(item.id)
                ev = TechEvent(
                    title=f"사건 {offset_days}",
                    occurred_at=T0 + timedelta(days=offset_days),
                    event_type="Launch",
                    merged_into_id=merged_into,
                )
                s.add(ev)
                await s.flush()
                event_ids.append(ev.id)
                s.add(DocumentEntity(tech_item_id=item.id, entity_id=via.id))
                s.add(EventDocument(event_id=ev.id, tech_item_id=item.id))
                await s.flush()
                return ev

            ns.a = await event(0, canon)
            ns.b = await event(1, alias)  # 별칭으로만 언급된 사건
            ns.c = await event(2, canon, merged_into=ns.b.id)  # 병합된 툼스톤
            await s.commit()
        ns.canon, ns.alias, ns.company, ns.bare = canon.id, alias.id, company.id, bare.id
        ns.sfx = sfx
        yield Session, ns
    finally:
        async with Session() as s:
            for ev_id in event_ids:  # 툼스톤(RESTRICT)부터
                ev = await s.get(TechEvent, ev_id)
                if ev is not None and ev.merged_into_id is not None:
                    await s.delete(ev)
                    await s.flush()
            for ev_id in event_ids:
                ev = await s.get(TechEvent, ev_id)
                if ev is not None:
                    await s.delete(ev)
                    await s.flush()
            for model, ids in ((TechItem, item_ids),):
                for i in ids:
                    obj = await s.get(model, i)
                    if obj is not None:
                        await s.delete(obj)
            await s.execute(delete(RoleBranch).where(RoleBranch.id.in_(branch_ids)))
            await s.execute(delete(Entity).where(Entity.id.in_(entity_ids)))
            await s.commit()
        await engine.dispose()


async def test_resolve_canonical_id_folds_alias() -> None:
    async with _seeded() as (Session, ns):
        async with Session() as s:
            assert await resolve_canonical_id(s, ns.alias) == ns.canon
            assert await resolve_canonical_id(s, ns.canon) == ns.canon
            assert await resolve_canonical_id(s, uuid.uuid4()) is None


async def test_is_technology_only_for_technology_verdict() -> None:
    async with _seeded() as (Session, ns):
        async with Session() as s:
            assert await is_technology(s, ns.canon) is True
            assert await is_technology(s, ns.company) is False
            assert await is_technology(s, ns.bare) is False


async def test_fetch_tech_page_full() -> None:
    async with _seeded() as (Session, ns):
        async with Session() as s:
            page = await fetch_tech_page(s, ns.canon)
        assert page is not None
        assert page.entity_id == ns.canon
        assert page.name
        assert page.description == "에이전트형 개인 비서"
        assert [b.name for b in page.branch_path] == [f"{TOP} {ns.sfx}", f"{LEAF} {ns.sfx}"]
        assert any("duet" in a.lower() and "claude" not in a.lower() for a in page.aliases)
        assert [e.id for e in page.events] == [ns.b.id, ns.a.id]
        assert page.events[0].event_type_label


async def test_fetch_tech_page_unclassified() -> None:
    async with _seeded(assign="none") as (Session, ns):
        async with Session() as s:
            page = await fetch_tech_page(s, ns.canon)
        assert page is not None and page.branch_path == ()
    async with _seeded(assign="null") as (Session, ns):
        async with Session() as s:
            page = await fetch_tech_page(s, ns.canon)
        assert page is not None and page.branch_path == ()


async def test_fetch_tech_page_not_technology_returns_none() -> None:
    async with _seeded() as (Session, ns):
        async with Session() as s:
            assert await fetch_tech_page(s, ns.company) is None
            assert await fetch_tech_page(s, ns.bare) is None


async def test_fetch_tech_page_empty_description_stays_empty() -> None:
    async with _seeded(description=None) as (Session, ns):
        async with Session() as s:
            page = await fetch_tech_page(s, ns.canon)
        assert page is not None and page.description is None


# --------------------------------------------------------------------- #
# ARG-319: 사건 기술 칩
# --------------------------------------------------------------------- #


async def test_event_chips_empty_input_returns_empty() -> None:
    assert await fetch_event_tech_chips(None, []) == {}  # type: ignore[arg-type]


@asynccontextmanager
async def _chip_seed():
    """사건 E1(대표 기사1 + 대표·별칭 같은 기사2 + 별칭만 기사3 + 회사), E2(기술 없음)."""
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.document_entity import DocumentEntity
    from argos.models.entity import Entity
    from argos.models.entity_alias import AliasRule, EntityAlias
    from argos.models.event_document import EventDocument
    from argos.models.tech_event import TechEvent
    from argos.models.tech_item import CategoryType, TechItem
    from argos.models.technology_profile import TechnologyProfile, TechVerdict

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    sfx = uuid.uuid4().hex[:8]
    ns = type("NS", (), {})()
    entity_ids: list[uuid.UUID] = []
    item_ids: list[uuid.UUID] = []
    event_ids: list[uuid.UUID] = []
    try:
        async with Session() as s:
            def ent(key: str, name: str) -> Entity:
                e = Entity(name=name, normalized_key=f"{key} {sfx}")
                s.add(e)
                return e

            big = ent("big", f"Zeta Big {sfx}")  # 대표, 사건 E1에서 기사 3건
            alias = ent("bigalias", f"Big Alias {sfx}")
            small = ent("small", f"Alpha Small {sfx}")  # 기사 1건
            tie = ent("tie", f"Beta Tie {sfx}")  # 기사 1건 -> small과 동률, 이름순
            company = ent("corp", f"Corp {sfx}")
            await s.flush()
            entity_ids += [big.id, alias.id, small.id, tie.id, company.id]
            s.add(EntityAlias(alias_entity_id=alias.id, canonical_entity_id=big.id, rule=AliasRule.MANUAL))
            for e in (big, small, tie):
                s.add(TechnologyProfile(entity_id=e.id, verdict=TechVerdict.TECHNOLOGY))
            s.add(TechnologyProfile(entity_id=company.id, verdict=TechVerdict.NOT_TECHNOLOGY))

            async def item() -> TechItem:
                it = TechItem(
                    title=f"doc {uuid.uuid4().hex[:6]}",
                    source_url=f"https://e.example/arg319/{uuid.uuid4()}",
                    raw_content="x",
                    category=CategoryType.MAINSTREAM,
                )
                s.add(it)
                await s.flush()
                item_ids.append(it.id)
                return it

            async def event(title: str) -> TechEvent:
                ev = TechEvent(title=title, occurred_at=T0, event_type="Launch")
                s.add(ev)
                await s.flush()
                event_ids.append(ev.id)
                return ev

            e1, e2 = await event("사건 1"), await event("사건 2")
            d1, d2, d3, d4 = await item(), await item(), await item(), await item()
            # d1: 대표 + 회사 / d2: 대표와 별칭이 같은 기사에 둘 다 / d3: 별칭만
            for it, ents in ((d1, [big, company, small]), (d2, [big, alias, tie]), (d3, [alias])):
                for e in ents:
                    s.add(DocumentEntity(tech_item_id=it.id, entity_id=e.id))
                s.add(EventDocument(event_id=e1.id, tech_item_id=it.id))
            s.add(DocumentEntity(tech_item_id=d4.id, entity_id=company.id))
            s.add(EventDocument(event_id=e2.id, tech_item_id=d4.id))
            await s.commit()
        ns.e1, ns.e2 = e1.id, e2.id
        ns.big, ns.alias, ns.small, ns.tie, ns.company = big.id, alias.id, small.id, tie.id, company.id
        yield Session, ns
    finally:
        async with Session() as s:
            await s.execute(delete(TechEvent).where(TechEvent.id.in_(event_ids)))
            await s.execute(delete(TechItem).where(TechItem.id.in_(item_ids)))
            await s.execute(delete(Entity).where(Entity.id.in_(entity_ids)))
            await s.commit()
        await engine.dispose()


async def test_event_chips_fold_alias_and_count_distinct_docs() -> None:
    async with _chip_seed() as (Session, ns):
        async with Session() as s:
            chips = await fetch_event_tech_chips(s, [ns.e1, ns.e2])
        top = chips[ns.e1][0]
        assert top.entity_id == ns.big  # 별칭이 대표로 접힌다
        assert top.doc_count == 3  # d1(대표) + d2(대표·별칭 둘 다 -> 1) + d3(별칭만)
        assert top.href == f"/tech/{ns.big}"
        assert all(c.entity_id != ns.alias for c in chips[ns.e1])


async def test_event_chips_exclude_non_technology() -> None:
    async with _chip_seed() as (Session, ns):
        async with Session() as s:
            chips = await fetch_event_tech_chips(s, [ns.e1])
        assert ns.company not in {c.entity_id for c in chips[ns.e1]}


async def test_event_chips_order_by_doc_count_then_name() -> None:
    async with _chip_seed() as (Session, ns):
        async with Session() as s:
            chips = await fetch_event_tech_chips(s, [ns.e1])
        assert [c.entity_id for c in chips[ns.e1]] == [ns.big, ns.small, ns.tie]
        assert [c.doc_count for c in chips[ns.e1]] == [3, 1, 1]


async def test_event_chips_event_without_tech_absent() -> None:
    async with _chip_seed() as (Session, ns):
        async with Session() as s:
            chips = await fetch_event_tech_chips(s, [ns.e1, ns.e2])
        assert ns.e2 not in chips and ns.e1 in chips


# --------------------------------------------------------------------- #
# ARG-320: 갈래 페이지
# --------------------------------------------------------------------- #


@asynccontextmanager
async def _branch_seed():
    """위 갈래 1 + 아래 갈래 둘(하나는 비어 있음). 기술 3개와 NotTechnology 배정 1개."""
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from argos.models.entity import Entity
    from argos.models.role_assignment import RoleAssignment
    from argos.models.role_branch import RoleBranch
    from argos.models.technology_profile import TechnologyProfile, TechVerdict

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    sfx = uuid.uuid4().hex[:8]
    ns = type("NS", (), {})()
    entity_ids: list[uuid.UUID] = []
    branch_ids: list[uuid.UUID] = []
    try:
        async with Session() as s:
            top = RoleBranch(name=f"위 {sfx}", description="위 설명", position=0)
            s.add(top)
            await s.flush()
            a = RoleBranch(name=f"A갈래 {sfx}", description="a", parent_id=top.id, position=0)
            b = RoleBranch(name=f"B갈래 {sfx}", description="b", parent_id=top.id, position=1)
            s.add_all([a, b])
            await s.flush()
            branch_ids += [top.id, a.id, b.id]

            async def tech(name: str, branch, verdict=TechVerdict.TECHNOLOGY):
                e = Entity(name=f"{name} {sfx}", normalized_key=f"{name.lower()} {sfx}")
                s.add(e)
                await s.flush()
                entity_ids.append(e.id)
                s.add(TechnologyProfile(entity_id=e.id, verdict=verdict))
                s.add(RoleAssignment(entity_id=e.id, branch_id=branch.id))
                return e

            ns.zed = await tech("zed", a)
            ns.alpha = await tech("Alpha", a)
            ns.beta = await tech("beta", b, TechVerdict.TECHNOLOGY)
            ns.company = await tech("acme", a, TechVerdict.NOT_TECHNOLOGY)
            await s.commit()
        ns.top, ns.a, ns.b, ns.sfx = top.id, a.id, b.id, sfx
        yield Session, ns
    finally:
        async with Session() as s:
            await s.execute(delete(RoleBranch).where(RoleBranch.id.in_(branch_ids)))
            await s.execute(delete(Entity).where(Entity.id.in_(entity_ids)))
            await s.commit()
        await engine.dispose()


async def test_leaf_branch_lists_its_technologies_by_name() -> None:
    async with _branch_seed() as (Session, ns):
        async with Session() as s:
            page = await fetch_branch_page(s, ns.a)
        assert page is not None
        assert page.parent is not None and page.parent.id == ns.top
        assert page.description == "a"
        assert len(page.groups) == 1 and page.groups[0].branch.id == ns.a
        assert [t.entity_id for t in page.groups[0].techs] == [ns.alpha.id, ns.zed.id]
        assert page.groups[0].techs[0].href == f"/tech/{ns.alpha.id}"


async def test_top_branch_groups_children() -> None:
    async with _branch_seed() as (Session, ns):
        async with Session() as s:
            page = await fetch_branch_page(s, ns.top)
        assert page is not None and page.parent is None
        assert [g.branch.id for g in page.groups] == [ns.a, ns.b]
        assert [t.entity_id for t in page.groups[0].techs] == [ns.alpha.id, ns.zed.id]
        assert [t.entity_id for t in page.groups[1].techs] == [ns.beta.id]


async def test_unknown_branch_returns_none() -> None:
    async with _branch_seed() as (Session, ns):
        async with Session() as s:
            assert await fetch_branch_page(s, uuid.uuid4()) is None


async def test_branch_with_no_technologies_has_empty_group() -> None:
    from sqlalchemy import delete

    from argos.models.role_assignment import RoleAssignment

    async with _branch_seed() as (Session, ns):
        async with Session() as s:
            await s.execute(delete(RoleAssignment).where(RoleAssignment.branch_id == ns.b))
            await s.commit()
            page = await fetch_branch_page(s, ns.b)
        assert page is not None
        assert len(page.groups) == 1 and page.groups[0].techs == ()
