"""ARG-327: '비슷한 역할' 읽기 서비스 — DB 기반(없으면 self-skip)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from argos.config import settings
from argos.models.technology_profile import TechVerdict
from argos.web.services.similar import SIMILAR_LIMIT, fetch_mixed_timeline, fetch_similar_techs
from argos.web.services.technology import TechRef
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url

pytestmark = [
    pytest.mark.skipif(
        not _db_reachable(_DB_URL),
        reason="pgvector DB not reachable — skipping ARG-327 DB-backed tests",
    ),
    pytest.mark.asyncio,
]

T0 = datetime(2099, 3, 1, tzinfo=timezone.utc)


@asynccontextmanager
async def _world():
    """갈래 둘(위/아래 + 다른 아래)과 기술·사건을 심는 도구를 주는 컨텍스트."""
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
    from argos.models.technology_profile import TechnologyProfile

    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    sfx = uuid.uuid4().hex[:8]
    entity_ids: list[uuid.UUID] = []
    branch_ids: list[uuid.UUID] = []
    item_ids: list[uuid.UUID] = []
    event_ids: list[uuid.UUID] = []

    class W:
        pass

    w = W()
    w.sfx = sfx
    w.AliasRule, w.EntityAlias = AliasRule, EntityAlias
    try:
        async with Session() as s:
            top = RoleBranch(name=f"위 {sfx}", description="위")
            s.add(top)
            await s.flush()
            leaf = RoleBranch(name=f"아래 {sfx}", description="아래", parent_id=top.id)
            other = RoleBranch(name=f"다른 {sfx}", description="다른", parent_id=top.id)
            s.add_all([leaf, other])
            await s.flush()
            branch_ids += [top.id, leaf.id, other.id]
            w.top, w.leaf, w.other = top, leaf, other

            async def tech(
                key, *, branch="leaf", verdict=TechVerdict.TECHNOLOGY, evidence=None, profile=True
            ):
                e = Entity(name=f"{key.title()} {sfx}", normalized_key=f"{key} {sfx}")
                s.add(e)
                await s.flush()
                entity_ids.append(e.id)
                if profile:
                    s.add(TechnologyProfile(entity_id=e.id, verdict=verdict))
                target = {"leaf": leaf.id, "other": other.id, "top": top.id, "null": None}[branch]
                s.add(
                    RoleAssignment(
                        entity_id=e.id,
                        branch_id=target,
                        evidence=evidence if evidence is not None else {},
                        unclassified_reason="x" if target is None else None,
                    )
                )
                await s.flush()
                return e

            async def alias_of(canon, key):
                e = Entity(name=f"{key.title()} {sfx}", normalized_key=f"{key} {sfx}")
                s.add(e)
                await s.flush()
                entity_ids.append(e.id)
                s.add(EntityAlias(alias_entity_id=e.id, canonical_entity_id=canon.id, rule=AliasRule.MANUAL))
                await s.flush()
                return e

            async def event(via, days, *, title="사건", merged_into=None):
                item = TechItem(
                    title=f"doc {uuid.uuid4().hex[:6]}",
                    source_url=f"https://e.example/arg327/{uuid.uuid4()}",
                    raw_content="x",
                    category=CategoryType.MAINSTREAM,
                )
                s.add(item)
                await s.flush()
                item_ids.append(item.id)
                ev = TechEvent(
                    title=title,
                    occurred_at=T0 + timedelta(days=days),
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

            w.tech, w.alias_of, w.event, w.s = tech, alias_of, event, s
            w.base = await tech("claude duet")
            yield Session, w
            await s.commit()
    finally:
        async with Session() as s:
            for ev_id in event_ids:
                ev = await s.get(TechEvent, ev_id)
                if ev is not None and ev.merged_into_id is not None:
                    await s.delete(ev)
                    await s.flush()
            for ev_id in event_ids:
                ev = await s.get(TechEvent, ev_id)
                if ev is not None:
                    await s.delete(ev)
                    await s.flush()
            for i in item_ids:
                obj = await s.get(TechItem, i)
                if obj is not None:
                    await s.delete(obj)
            await s.execute(delete(RoleBranch).where(RoleBranch.id.in_(branch_ids)))
            await s.execute(delete(Entity).where(Entity.id.in_(entity_ids)))
            await s.commit()
        await engine.dispose()


def _neighbor(name: str, count: int, examples=("예문",)):
    return {"neighbors": [{"name": name, "count": count, "examples": list(examples)}]}


async def test_same_leaf_technology_only():
    async with _world() as (Session, w):
        good = await w.tech("alpha")
        await w.tech("beta", branch="other")
        await w.tech("gamma", verdict=TechVerdict.NOT_TECHNOLOGY)
        aliased = await w.tech("delta")
        await w.alias_of(good, "alpha2")
        canon = await w.tech("epsilon")
        # delta는 epsilon의 별칭 행을 가진다 -> 후보 아님
        w.s.add(w.EntityAlias(alias_entity_id=aliased.id, canonical_entity_id=canon.id, rule=w.AliasRule.MANUAL))
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert {t.entity_id for t in got} == {good.id, canon.id}
        assert all(t.branch.id == w.leaf.id for t in got)
        assert all(t.href == f"/tech/{t.entity_id}" for t in got)


async def test_unclassified_returns_empty():
    async with _world() as (Session, w):
        lone = await w.tech("lonely", branch="null")
        await w.tech("alpha")
        await w.s.commit()
        async with Session() as s:
            assert await fetch_similar_techs(s, lone.id) == ()
            # 배정 행 자체가 없는 엔티티도 ()
            assert await fetch_similar_techs(s, uuid.uuid4()) == ()


async def test_top_level_branch_returns_empty():
    async with _world() as (Session, w):
        top = await w.tech("toplevel", branch="top")
        await w.tech("sibling", branch="top")
        await w.s.commit()
        async with Session() as s:
            assert await fetch_similar_techs(s, top.id) == ()


async def test_alias_and_contained_name_excluded():
    async with _world() as (Session, w):
        duet = await w.tech("duet")  # claude duet 안에 낱말로 포함 -> 제외
        go = await w.tech("go")
        django = await w.tech("django")  # 낱말이 아니라 글자 포함 -> 유지
        # 별칭 키가 기준 이름을 포함하는 후보도 제외
        wrapped = await w.tech("wrapper")
        await w.alias_of(wrapped, "pro claude duet")
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        ids = {t.entity_id for t in got}
        assert duet.id not in ids
        assert wrapped.id not in ids
        assert {go.id, django.id} <= ids


async def test_limit_is_five():
    async with _world() as (Session, w):
        for i in range(SIMILAR_LIMIT + 3):
            await w.tech(f"tool{i}")
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert len(got) == SIMILAR_LIMIT == 5


async def test_order_evidence_then_recent():
    async with _world() as (Session, w):
        listed = await w.tech("listed", evidence=_neighbor(f"claude duet {w.sfx}", 4))
        await w.event(listed, 1)  # 옛 사건
        fresh = await w.tech("fresh")
        await w.event(fresh, 50)  # 근거 없는 최신
        older = await w.tech("older")
        await w.event(older, 10)
        silent = await w.tech("silent")  # 사건 없음 -> 맨 뒤
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert [t.entity_id for t in got] == [listed.id, fresh.id, older.id, silent.id]


async def test_latest_event_skips_tombstone_and_follows_alias():
    async with _world() as (Session, w):
        cand = await w.tech("candidate")
        al = await w.alias_of(cand, "candi")
        via_canon = await w.event(cand, 1, title="대표로 언급")
        newest = await w.event(al, 3, title="별칭으로 언급")  # 별칭 언급도 대표로 접힌다
        await w.event(cand, 9, title="툼스톤", merged_into=newest.id)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        (only,) = [t for t in got if t.entity_id == cand.id]
        assert only.latest_event.id == newest.id
        assert only.latest_event.title == "별칭으로 언급"
        assert via_canon.id != only.latest_event.id


async def test_no_event_still_listed():
    async with _world() as (Session, w):
        cand = await w.tech("eventless")
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        (only,) = got
        assert only.entity_id == cand.id and only.latest_event is None
        assert only.colisted_count == 0 and only.colisted_example is None


async def test_neighbor_alias_name_folds_to_candidate():
    async with _world() as (Session, w):
        base_ev = {"neighbors": [{"name": f"cand alt {w.sfx}", "count": 3, "examples": ["함께 쓰임"]}]}
        from sqlalchemy import update

        from argos.models.role_assignment import RoleAssignment

        await w.s.execute(
            update(RoleAssignment).where(RoleAssignment.entity_id == w.base.id).values(evidence=base_ev)
        )
        cand = await w.tech("candidate")
        await w.alias_of(cand, "cand alt")
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        (only,) = got
        assert only.colisted_count == 3
        assert only.colisted_example == "함께 쓰임"


async def test_evidence_from_candidate_row_counts():
    async with _world() as (Session, w):
        cand = await w.tech(
            "candidate",
            evidence={"neighbors": [
                {"name": f"claude duet {w.sfx}", "count": 2, "examples": ["작은"]},
                {"name": f"claude duet {w.sfx}", "count": 7, "examples": ["큰", "둘째"]},
                {"name": "unrelated", "count": 99, "examples": ["무관"]},
            ]},
        )
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        (only,) = got
        assert only.entity_id == cand.id
        assert only.colisted_count == 7 and only.colisted_example == "큰"


async def test_malformed_evidence_ignored():
    bad = [
        {"neighbors": "nope"},
        {"neighbors": [1, None, "x", {"name": 5, "count": 2}, {"name": "a", "count": "2"},
                       {"name": "a", "count": True}, {"name": "a", "count": -1}]},
        {"neighbors": {"name": "a"}},
        {"other": 1},
        {},
    ]
    async with _world() as (Session, w):
        for i, ev in enumerate(bad):
            await w.tech(f"bad{i}", evidence=ev)
        # 예문 모양이 이상해도 count가 맞으면 근거는 인정, 예문은 None
        weird = await w.tech(
            "weird", evidence={"neighbors": [{"name": f"claude duet {w.sfx}", "count": 2, "examples": "x"}]}
        )
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        by_id = {t.entity_id: t for t in got}
        assert by_id[weird.id].colisted_count == 2 and by_id[weird.id].colisted_example is None
        assert all(t.colisted_count == 0 for t in got if t.entity_id != weird.id)


# --- ARG-328: 타임라인 섞기 -------------------------------------------------


def _ref(e) -> TechRef:
    return TechRef(entity_id=e.id, name=e.name)


async def _similar_of(Session, w):
    async with Session() as s:
        return await fetch_similar_techs(s, w.base.id)


async def test_mixed_order_is_chronological():
    async with _world() as (Session, w):
        a = await w.tech("alpha")
        old = await w.event(w.base, 1, title="내 옛 사건")
        mid = await w.event(a, 5, title="비슷한 기술 사건")
        new = await w.event(w.base, 9, title="내 새 사건")
        await w.s.commit()
        similar = await _similar_of(Session, w)
        async with Session() as s:
            got = await fetch_mixed_timeline(s, _ref(w.base), similar)
        assert [e.id for e in got] == [new.id, mid.id, old.id]


async def test_event_mentioning_both_appears_once_with_both_names():
    async with _world() as (Session, w):
        from sqlalchemy import select

        from argos.models.document_entity import DocumentEntity
        from argos.models.event_document import EventDocument

        a = await w.tech("alpha")
        ev = await w.event(w.base, 2, title="둘 다 나온 사건")
        item_id = (
            await w.s.execute(select(EventDocument.tech_item_id).where(EventDocument.event_id == ev.id))
        ).scalar_one()
        w.s.add(DocumentEntity(tech_item_id=item_id, entity_id=a.id))
        await w.s.commit()
        similar = await _similar_of(Session, w)
        async with Session() as s:
            got = await fetch_mixed_timeline(s, _ref(w.base), similar)
        assert [e.id for e in got] == [ev.id]
        # 이 기술이 먼저, 그다음 비슷한 기술.
        assert [t.entity_id for t in got[0].techs] == [w.base.id, a.id]
        assert got[0].event_type_label == "출시"


async def test_mixed_folds_alias_and_skips_tombstone():
    async with _world() as (Session, w):
        a = await w.tech("alpha")
        al = await w.alias_of(a, "alpha2")
        via_alias = await w.event(al, 3, title="별칭으로 언급된 사건")
        survivor = await w.event(w.base, 4, title="살아남은 사건")
        await w.event(a, 6, title="흡수된 사건", merged_into=survivor.id)
        await w.s.commit()
        similar = await _similar_of(Session, w)
        async with Session() as s:
            got = await fetch_mixed_timeline(s, _ref(w.base), similar)
        assert [e.id for e in got] == [survivor.id, via_alias.id]
        assert [t.entity_id for t in got[1].techs] == [a.id]
