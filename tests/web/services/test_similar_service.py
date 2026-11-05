"""ARG-327: '비슷한 역할' 읽기 서비스 — DB 기반(없으면 self-skip)."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from argos.config import settings
from argos.models.technology_profile import TechVerdict
from argos.web.services.similar import (
    SIMILAR_LIMIT,
    SIMILAR_NEWS_LIMIT,
    SimilarNews,
    fetch_event_similar_news,
    fetch_mixed_timeline,
    fetch_similar_techs,
)
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
        techs = []
        for i in range(SIMILAR_LIMIT + 3):
            t = await w.tech(f"tool{i}")
            await w.event(t, i)  # 클수록 최근 -> 위쪽 순위
            techs.append(t)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert len(got) == SIMILAR_LIMIT == 5
        # 순위 위 다섯만 남고, 가장 낮은 순위들이 잘린다.
        expected = [t.id for t in reversed(techs)][:SIMILAR_LIMIT]
        assert [t.entity_id for t in got] == expected
        assert techs[0].id not in {t.entity_id for t in got}


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


async def _also_mention(w, ev, tech):
    """이미 있는 사건의 기사에 기술 하나를 더 언급시킨다."""
    from sqlalchemy import select

    from argos.models.document_entity import DocumentEntity
    from argos.models.event_document import EventDocument

    item_id = (
        await w.s.execute(select(EventDocument.tech_item_id).where(EventDocument.event_id == ev.id))
    ).scalar_one()
    w.s.add(DocumentEntity(tech_item_id=item_id, entity_id=tech.id))
    await w.s.flush()


async def test_one_row_per_latest_event():
    """한 사건에 함께 나온 이름들이 목록을 채우지 않는다 — 순위가 위인 하나만 남는다."""
    async with _world() as (Session, w):
        alpha = await w.tech("alpha")
        beta = await w.tech("beta")
        ev = await w.event(alpha, 9, title="함께 나온 사건")
        await _also_mention(w, ev, beta)
        solo = await w.tech("solo")
        await w.event(solo, 1)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert [t.entity_id for t in got] == [alpha.id, solo.id]


async def test_dropped_repeats_open_slots_for_lower_candidates():
    """같은 사건으로 빠진 자리는 다음 후보가 채운다 — 상한은 뺀 뒤에 적용한다."""
    async with _world() as (Session, w):
        crowd = [await w.tech(f"crowd{i}") for i in range(SIMILAR_LIMIT)]
        ev = await w.event(crowd[0], 50)
        for t in crowd[1:]:
            await _also_mention(w, ev, t)
        rest = []
        for i in range(SIMILAR_LIMIT):
            t = await w.tech(f"rest{i}")
            await w.event(t, i)
            rest.append(t)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert len(got) == SIMILAR_LIMIT
        assert len({t.latest_event.id for t in got}) == SIMILAR_LIMIT


async def test_candidates_without_event_are_not_collapsed():
    async with _world() as (Session, w):
        a = await w.tech("silent a")
        b = await w.tech("silent b")
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert {t.entity_id for t in got} == {a.id, b.id}


async def test_no_evidence_prefers_mentions_over_recency():
    async with _world() as (Session, w):
        fresh = await w.tech("fresh")
        await w.event(fresh, 50)
        popular = await w.tech("popular")
        for day in (1, 2, 3):
            await w.event(popular, day)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert [t.entity_id for t in got] == [popular.id, fresh.id]


async def test_no_evidence_prefers_events_shared_with_base():
    """같은 갈래라도 기준 기술마다 목록이 달라지는 신호 — 언급 수보다 먼저 본다."""
    async with _world() as (Session, w):
        popular = await w.tech("popular")
        for day in (1, 2, 3, 4):
            await w.event(popular, day)
        partner = await w.tech("partner")
        shared = await w.event(partner, 5)
        await _also_mention(w, shared, w.base)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert [t.entity_id for t in got] == [partner.id, popular.id]


async def test_same_branch_techs_get_different_lists_from_shared_events():
    async with _world() as (Session, w):
        other_base = await w.tech("other base")
        for_base = await w.tech("for base")
        await _also_mention(w, await w.event(for_base, 1), w.base)
        for_other = await w.tech("for other")
        await _also_mention(w, await w.event(for_other, 2), other_base)
        await w.s.commit()
        async with Session() as s:
            mine = await fetch_similar_techs(s, w.base.id)
        async with Session() as s:
            theirs = await fetch_similar_techs(s, other_base.id)
        assert mine[0].entity_id == for_base.id
        assert theirs[0].entity_id == for_other.id


async def test_evidence_candidates_stay_ahead_and_ordered_by_recency():
    """근거 있는 후보는 언급이 적어도 먼저이고, 그 안에서는 여전히 최근 순이다."""
    async with _world() as (Session, w):
        key = f"claude duet {w.sfx}"
        popular = await w.tech("popular")
        for day in (1, 2, 3, 4):
            await w.event(popular, day)
        many = await w.tech("many", evidence=_neighbor(key, 2))
        for day in (5, 6, 7):
            await w.event(many, day)
        recent = await w.tech("recent", evidence=_neighbor(key, 2))
        await w.event(recent, 8)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        assert [t.entity_id for t in got] == [recent.id, many.id, popular.id]


async def test_mentions_count_events_folds_alias_and_skips_tombstone():
    """언급 수는 사건 수다 — 같은 사건의 기사 여러 건은 1, 별칭은 대표로, 툼스톤은 제외."""
    async with _world() as (Session, w):
        fresh = await w.tech("fresh")
        await w.event(fresh, 50)
        await w.event(fresh, 49)

        folded = await w.tech("folded")
        alias = await w.alias_of(folded, "folded nick")
        await w.event(folded, 1)
        await w.event(alias, 2)
        await w.event(alias, 3)

        ghost = await w.tech("ghost")
        survivor = await w.event(ghost, 4)
        await w.event(ghost, 5, merged_into=survivor.id)
        await w.event(ghost, 6, merged_into=survivor.id)
        await w.s.commit()
        async with Session() as s:
            got = await fetch_similar_techs(s, w.base.id)
        # folded 3건 > fresh 2건 > ghost 1건(툼스톤 2건 제외)
        assert [t.entity_id for t in got] == [folded.id, fresh.id, ghost.id]


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


async def _per_chip_news(session, event_id, chip_ids):
    """칩마다 ``fetch_similar_techs``를 따로 불러 같은 규칙으로 합친 기준 결과."""
    chip_set = set(chip_ids)
    seen_techs: set = set()
    seen_events = {event_id}
    found = []
    for cid in chip_ids:
        for t in await fetch_similar_techs(session, cid):
            ev = t.latest_event
            if t.entity_id in chip_set or t.entity_id in seen_techs:
                continue
            if ev is None or ev.id in seen_events:
                continue
            seen_techs.add(t.entity_id)
            seen_events.add(ev.id)
            found.append(SimilarNews(entity_id=t.entity_id, name=t.name, branch=t.branch, event=ev))
    found.sort(key=lambda n: (n.event.occurred_at, str(n.event.id)), reverse=True)
    return tuple(found[:SIMILAR_NEWS_LIMIT])


async def _count_statements(Session, fn):
    from sqlalchemy import event as sa_event

    sync_engine = Session.kw["bind"].sync_engine
    count = 0

    def _before(conn, cursor, statement, parameters, context, executemany):
        nonlocal count
        count += 1

    sa_event.listen(sync_engine, "before_cursor_execute", _before)
    try:
        async with Session() as s:
            result = await fn(s)
    finally:
        sa_event.remove(sync_engine, "before_cursor_execute", _before)
    return count, result


async def test_event_news_statements_do_not_scale_with_chips_in_one_branch():
    async with _world() as (Session, w):
        for i in range(8):
            cand = await w.tech(f"cand{i}")
            await w.event(cand, i)
        chips = [w.base] + [await w.tech(f"chip{i}") for i in range(5)]
        await w.s.commit()
        ids = [c.id for c in chips]
        this_event = uuid.uuid4()

        n2, _ = await _count_statements(
            Session, lambda s: fetch_event_similar_news(s, this_event, ids[:2])
        )
        n6, _ = await _count_statements(
            Session, lambda s: fetch_event_similar_news(s, this_event, ids)
        )
        assert n6 <= n2 + 1, (n2, n6)
        # 칩마다 따로 부르던 때(칩 하나당 5쿼리 안팎)보다 훨씬 적다.
        assert n6 < 6 * 3, n6


async def test_event_news_equals_per_chip_lookup():
    async with _world() as (Session, w):
        listed = await w.tech("listed", evidence=_neighbor(f"claude duet {w.sfx}", 4))
        await w.event(listed, 1)
        for i in range(5):
            c = await w.tech(f"cand{i}")
            await w.event(c, 10 + i)
        elsewhere = await w.tech("far", branch="other")
        await w.event(elsewhere, 99)
        other_chip = await w.tech("otherchip", branch="other")
        chip_in_leaf = await w.tech("leafchip")
        unclassified = await w.tech("unclass", branch="null")
        await w.s.commit()
        ids = [w.base.id, other_chip.id, chip_in_leaf.id, unclassified.id, uuid.uuid4()]
        this_event = uuid.uuid4()
        async with Session() as s:
            merged = await fetch_event_similar_news(s, this_event, ids)
        async with Session() as s:
            expected = await _per_chip_news(s, this_event, ids)
        assert merged == expected
        assert merged  # 비어 있으면 동등성 검증이 무의미하다
