"""사건 경계 교정 반영 통합 검증 (ARG-245). DB가 없으면 통째로 skip한다."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain import event_correction
from argos.brain.event_correction import (
    SKIP_CAPPED,
    SKIP_CONFLICT,
    SKIP_DOCUMENTS_CHANGED,
    SKIP_INVALID_FRAGMENTS,
    SKIP_TOMBSTONED,
    SKIP_UNRESOLVABLE,
    apply_corrections,
)
from argos.brain.recluster_candidates import MergeCandidate, ReclusterCandidates, SplitCandidate
from argos.config import settings
from argos.models.entity import Entity, EventEntity
from argos.models.event_document import EventDocument
from argos.models.tech_event import TechEvent
from argos.models.tech_item import CategoryType, TechItem
from argos.services.event_resolution import resolve_event
from argos.services.events import list_evidence_documents
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-245-correction-test.example.com/"
_TITLE_PREFIX = "ARG-245 "
_ENTITY_PREFIX = "arg-245-"
_BASE = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping ARG-245 correction DB tests")


@pytest.fixture(scope="module")
def session_factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    yield async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def clean(session_factory):
    async def _wipe():
        async with session_factory() as session:
            item_ids = (
                await session.execute(
                    select(TechItem.id).where(TechItem.source_url.like(f"{_URL_PREFIX}%"))
                )
            ).scalars().all()
            by_title = set(
                (
                    await session.execute(
                        select(TechEvent.id).where(TechEvent.title.like(f"{_TITLE_PREFIX}%"))
                    )
                ).scalars()
            )
            # 분할이 만든 사건은 title이 None이라 제목으로는 못 찾는다 — 우리 문서에 붙은 사건도 모은다.
            by_link = set(
                (
                    await session.execute(
                        select(EventDocument.event_id).where(
                            EventDocument.tech_item_id.in_(item_ids or [uuid.uuid4()])
                        )
                    )
                ).scalars()
            )
            event_ids = list(by_title | by_link)
            if event_ids:
                await session.execute(delete(EventEntity).where(EventEntity.event_id.in_(event_ids)))
                await session.execute(delete(EventDocument).where(EventDocument.event_id.in_(event_ids)))
                # 툼스톤 포인터부터 끊어야 RESTRICT FK에 막히지 않는다.
                await session.execute(
                    TechEvent.__table__.update()
                    .where(TechEvent.id.in_(event_ids))
                    .values(merged_into_id=None)
                )
                await session.execute(delete(TechEvent).where(TechEvent.id.in_(event_ids)))
            if item_ids:
                await session.execute(delete(TechItem).where(TechItem.id.in_(item_ids)))
            await session.execute(delete(Entity).where(Entity.normalized_key.like(f"{_ENTITY_PREFIX}%")))
            await session.commit()

    await _wipe()
    yield
    await _wipe()


async def _event(session, name: str, *, at: datetime = _BASE, merged_into=None) -> TechEvent:
    event = TechEvent(title=f"{_TITLE_PREFIX}{name}", occurred_at=at, merged_into_id=merged_into)
    session.add(event)
    await session.flush()
    return event


async def _doc(session, name: str, *, at: datetime = _BASE) -> TechItem:
    item = TechItem(
        title=f"{_TITLE_PREFIX}{name}",
        source_url=f"{_URL_PREFIX}{name}",
        raw_content="x",
        category=CategoryType.MAINSTREAM,
        published_at=at,
    )
    session.add(item)
    await session.flush()
    return item


async def _entity(session, name: str) -> Entity:
    entity = Entity(name=name, normalized_key=f"{_ENTITY_PREFIX}{name}")
    session.add(entity)
    await session.flush()
    return entity


async def _links(session, event_id) -> set[uuid.UUID]:
    return set(
        (
            await session.execute(
                select(EventDocument.tech_item_id).where(EventDocument.event_id == event_id)
            )
        ).scalars()
    )


async def _entity_links(session, event_id) -> set[uuid.UUID]:
    return set(
        (
            await session.execute(
                select(EventEntity.entity_id).where(EventEntity.event_id == event_id)
            )
        ).scalars()
    )


@pytest.mark.asyncio
async def test_cleanup_moves_tombstone_links_to_final_survivor(session_factory, clean):
    """툼스톤 사슬 A→B→C에서 A·B에 남은 문서·엔티티는 C로 간다. 중복은 한 번만."""
    async with session_factory() as session:
        c = await _event(session, "c")
        b = await _event(session, "b", merged_into=c.id)
        a = await _event(session, "a", merged_into=b.id)
        d1, d2, d_shared = await _doc(session, "d1"), await _doc(session, "d2"), await _doc(session, "shared")
        e1 = await _entity(session, "e1")
        session.add_all([
            EventDocument(event_id=a.id, tech_item_id=d1.id),
            EventDocument(event_id=b.id, tech_item_id=d2.id),
            EventDocument(event_id=b.id, tech_item_id=d_shared.id),
            EventDocument(event_id=c.id, tech_item_id=d_shared.id),
            EventEntity(event_id=a.id, entity_id=e1.id),
        ])
        await session.commit()

    result = await apply_corrections(session_factory)

    async with session_factory() as session:
        assert await _links(session, c.id) == {d1.id, d2.id, d_shared.id}
        assert await _links(session, a.id) == set()
        assert await _links(session, b.id) == set()
        assert await _entity_links(session, c.id) == {e1.id}
        survivor = await session.get(TechEvent, c.id)
        assert survivor.naming_stale is True
        # 사건은 하나도 지워지지 않는다
        assert (await session.get(TechEvent, a.id)).merged_into_id == b.id

    ours = [r for r in result.relocations if r.to_event_id == c.id]
    assert {(r.kind, r.target_id, r.deduplicated) for r in ours} == {
        ("document", d1.id, False),
        ("document", d2.id, False),
        ("document", d_shared.id, True),
        ("entity", e1.id, False),
    }


@pytest.mark.asyncio
async def test_cleanup_skips_chain_that_never_reaches_a_live_event(session_factory, clean):
    """순환 사슬(x→y→x)은 살아 있는 사건이 없다 — 옮기지 않고 사유를 남긴다."""
    async with session_factory() as session:
        x = await _event(session, "x")
        y = await _event(session, "y", merged_into=x.id)
        await session.flush()
        x.merged_into_id = y.id
        d = await _doc(session, "cyc")
        session.add(EventDocument(event_id=x.id, tech_item_id=d.id))
        await session.commit()

    result = await apply_corrections(session_factory)

    async with session_factory() as session:
        assert await _links(session, x.id) == {d.id}
    assert any(
        s.kind == "cleanup" and s.reason == SKIP_UNRESOLVABLE and x.id in s.event_ids
        for s in result.skipped
    )


@pytest.mark.asyncio
async def test_failure_rolls_back_every_cleanup_write(session_factory, clean, monkeypatch):
    """두 번째 툼스톤의 문서 이동에서 터지면 첫 번째 툼스톤의 이동도 남지 않는다.

    `_move_links` 호출 순서: 첫 툼스톤 문서(1) → 첫 툼스톤 엔티티(2) → 둘째
    툼스톤 문서(3). 세 번째 호출에서 터뜨려야 툼스톤을 넘는 롤백을 본다.
    """
    async with session_factory() as session:
        survivor = await _event(session, "survivor")
        t1 = await _event(session, "t1", merged_into=survivor.id)
        t2 = await _event(session, "t2", merged_into=survivor.id)
        d1, d2 = await _doc(session, "r1"), await _doc(session, "r2")
        session.add_all([
            EventDocument(event_id=t1.id, tech_item_id=d1.id),
            EventDocument(event_id=t2.id, tech_item_id=d2.id),
        ])
        await session.commit()

    real_move = event_correction._move_links
    calls = {"n": 0}

    async def flaky_move(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("boom")
        return await real_move(*args, **kwargs)

    monkeypatch.setattr(event_correction, "_move_links", flaky_move)
    with pytest.raises(RuntimeError, match="boom"):
        await apply_corrections(session_factory)

    async with session_factory() as session:
        assert await _links(session, t1.id) == {d1.id}
        assert await _links(session, t2.id) == {d2.id}
        assert await _links(session, survivor.id) == set()
        assert (await session.get(TechEvent, survivor.id)).naming_stale is False


def _merges(*pairs_with_evidence) -> ReclusterCandidates:
    return ReclusterCandidates(
        merges=tuple(
            MergeCandidate(event_ids=tuple(sorted(pair)), evidence_document_ids=tuple(sorted(ev)))
            for pair, ev in pairs_with_evidence
        ),
        splits=(),
    )


@pytest.mark.asyncio
async def test_merge_chain_tombstones_absorbed_and_sums_sources(session_factory, clean):
    """A–B, B–C 후보 → 한 그룹. 문서가 가장 많은 B가 생존, A·C는 B를 직접 가리킨다."""
    async with session_factory() as session:
        a = await _event(session, "ma", at=_BASE)
        b = await _event(session, "mb", at=_BASE + timedelta(days=1))
        c = await _event(session, "mc", at=_BASE + timedelta(days=2))
        old = await _event(session, "old", merged_into=a.id)  # A를 가리키던 기존 툼스톤
        da, db1, db2, dc, dshared = [await _doc(session, n) for n in ("da", "db1", "db2", "dc", "dsh")]
        ent = await _entity(session, "ment")
        session.add_all([
            EventDocument(event_id=a.id, tech_item_id=da.id),
            EventDocument(event_id=b.id, tech_item_id=db1.id),
            EventDocument(event_id=b.id, tech_item_id=db2.id),
            EventDocument(event_id=b.id, tech_item_id=dshared.id),
            EventDocument(event_id=c.id, tech_item_id=dc.id),
            EventDocument(event_id=c.id, tech_item_id=dshared.id),
            EventEntity(event_id=c.id, entity_id=ent.id),
        ])
        await session.commit()

    result = await apply_corrections(
        session_factory,
        _merges(((a.id, b.id), (da.id, db1.id)), ((b.id, c.id), (db2.id, dc.id))),
    )

    assert len(result.merges) == 1
    merge = result.merges[0]
    assert merge.survivor_id == b.id
    assert merge.absorbed_ids == tuple(sorted((a.id, c.id)))
    assert merge.document_count == 5  # da, db1, db2, dc, dshared — dshared는 한 번만

    async with session_factory() as session:
        survivor = await session.get(TechEvent, b.id)
        assert survivor.merged_into_id is None
        assert survivor.naming_stale is True
        assert survivor.occurred_at == _BASE  # 병합된 사건 중 가장 이른 시각
        for absorbed in (a.id, c.id, old.id):
            row = await session.get(TechEvent, absorbed)
            assert row is not None  # 삭제 금지
            assert row.merged_into_id == b.id  # 사슬 압축: 최종 생존자를 직접
        assert await _links(session, b.id) == {da.id, db1.id, db2.id, dc.id, dshared.id}
        assert await _links(session, a.id) == set() and await _links(session, c.id) == set()
        assert await _entity_links(session, b.id) == {ent.id}
        # 옛 링크(서비스 수준): 흡수된 id로 들어와도 생존자 내용이 보인다
        for old_link in (a.id, c.id, old.id):
            resolved = await resolve_event(session, old_link)
            assert resolved == b.id
            docs = await list_evidence_documents(session, resolved)
            assert {d.id for d in docs} == {da.id, db1.id, db2.id, dc.id, dshared.id}


@pytest.mark.asyncio
async def test_stale_merge_groups_are_skipped_and_others_still_apply(session_factory, clean):
    async with session_factory() as session:
        live = await _event(session, "live")
        gone = await _event(session, "gone", merged_into=live.id)  # 이미 툼스톤
        p = await _event(session, "p")
        q = await _event(session, "q")
        r = await _event(session, "r")
        s = await _event(session, "s")
        dp, dq, dr, ds, dmoved = [await _doc(session, n) for n in ("dp", "dq", "dr", "ds", "dmv")]
        session.add_all([
            EventDocument(event_id=p.id, tech_item_id=dp.id),
            EventDocument(event_id=q.id, tech_item_id=dq.id),
            EventDocument(event_id=r.id, tech_item_id=dr.id),
            EventDocument(event_id=s.id, tech_item_id=ds.id),
        ])
        await session.commit()

    result = await apply_corrections(
        session_factory,
        _merges(
            ((gone.id, p.id), (dp.id,)),           # 낡음: gone이 이미 툼스톤
            ((q.id, r.id), (dq.id, dmoved.id)),    # 낡음: dmoved가 어느 쪽에도 안 붙음
            ((s.id, live.id), (ds.id,)),           # 정상
        ),
    )

    reasons = {(s_.kind, s_.reason, s_.event_ids) for s_ in result.skipped}
    assert ("merge", SKIP_TOMBSTONED, tuple(sorted((gone.id, p.id)))) in reasons
    assert ("merge", SKIP_DOCUMENTS_CHANGED, tuple(sorted((q.id, r.id)))) in reasons
    assert [m.absorbed_ids for m in result.merges] == [(live.id,)]
    assert result.merges[0].survivor_id == s.id  # s 문서 1건 > live 0건
    async with session_factory() as session:
        assert (await session.get(TechEvent, p.id)).merged_into_id is None
        assert (await session.get(TechEvent, q.id)).merged_into_id is None
        assert (await session.get(TechEvent, r.id)).merged_into_id is None


@pytest.mark.asyncio
async def test_merge_failure_rolls_back_everything(session_factory, clean, monkeypatch):
    async with session_factory() as session:
        a = await _event(session, "fa")
        b = await _event(session, "fb")
        da, db = await _doc(session, "fda"), await _doc(session, "fdb")
        session.add_all([
            EventDocument(event_id=a.id, tech_item_id=da.id),
            EventDocument(event_id=b.id, tech_item_id=db.id),
        ])
        await session.commit()

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(event_correction, "_mark_naming_stale", boom)
    with pytest.raises(RuntimeError):
        await apply_corrections(session_factory, _merges(((a.id, b.id), (da.id, db.id))))

    async with session_factory() as session:
        assert (await session.get(TechEvent, a.id)).merged_into_id is None
        assert (await session.get(TechEvent, b.id)).merged_into_id is None
        assert await _links(session, a.id) == {da.id}
        assert await _links(session, b.id) == {db.id}


@pytest.mark.asyncio
async def test_split_keeps_id_on_largest_fragment_and_creates_new_events(session_factory, clean):
    async with session_factory() as session:
        ev = await _event(session, "split", at=_BASE)
        other = await _event(session, "other")
        big = [await _doc(session, f"big{i}", at=_BASE + timedelta(hours=i)) for i in range(3)]
        small = [await _doc(session, f"small{i}", at=_BASE + timedelta(days=2, hours=i)) for i in range(2)]
        outside = await _doc(session, "outside", at=_BASE - timedelta(days=30))
        ent = await _entity(session, "sent")
        for d in big + small + [outside]:
            session.add(EventDocument(event_id=ev.id, tech_item_id=d.id))
        session.add(EventDocument(event_id=other.id, tech_item_id=small[0].id))  # N:N 링크
        session.add(EventEntity(event_id=ev.id, entity_id=ent.id))
        await session.commit()

    candidates = ReclusterCandidates(
        merges=(),
        splits=(
            SplitCandidate(
                event_id=ev.id,
                groups=(tuple(sorted(d.id for d in small)), tuple(sorted(d.id for d in big))),
            ),
        ),
    )
    result = await apply_corrections(session_factory, candidates)

    assert len(result.splits) == 1
    applied = result.splits[0]
    assert applied.event_id == ev.id
    assert len(applied.new_event_ids) == 1
    new_id = applied.new_event_ids[0]
    async with session_factory() as session:
        assert await _links(session, ev.id) == {d.id for d in big} | {outside.id}
        assert await _links(session, new_id) == {d.id for d in small}
        assert await _links(session, other.id) == {small[0].id}  # 다른 사건 링크는 그대로
        new_event = await session.get(TechEvent, new_id)
        assert new_event.title is None and new_event.summary is None
        assert new_event.naming_stale is True
        assert new_event.occurred_at == _BASE + timedelta(days=2)
        assert (await session.get(TechEvent, ev.id)).naming_stale is True
        assert await _entity_links(session, ev.id) == {ent.id}
        assert await _entity_links(session, new_id) == set()


@pytest.mark.asyncio
async def test_capped_conflicting_and_stale_splits_are_skipped(session_factory, clean):
    async with session_factory() as session:
        capped = await _event(session, "capped")
        conflict = await _event(session, "conflict")
        partner = await _event(session, "partner")
        stale = await _event(session, "stale")
        docs = {n: await _doc(session, n) for n in ("c1", "c2", "k1", "k2", "pk", "s1", "s2")}
        for n in ("c1", "c2"):
            session.add(EventDocument(event_id=capped.id, tech_item_id=docs[n].id))
        for n in ("k1", "k2"):
            session.add(EventDocument(event_id=conflict.id, tech_item_id=docs[n].id))
        session.add(EventDocument(event_id=partner.id, tech_item_id=docs["pk"].id))
        session.add(EventDocument(event_id=stale.id, tech_item_id=docs["s1"].id))  # s2는 안 붙음
        await session.commit()

    candidates = ReclusterCandidates(
        merges=(
            MergeCandidate(
                event_ids=tuple(sorted((conflict.id, partner.id))),
                evidence_document_ids=tuple(sorted((docs["k1"].id, docs["pk"].id))),
            ),
        ),
        splits=(
            SplitCandidate(event_id=capped.id, groups=((docs["c1"].id,), (docs["c2"].id,)), capped=True),
            SplitCandidate(event_id=conflict.id, groups=((docs["k1"].id,), (docs["k2"].id,))),
            SplitCandidate(event_id=stale.id, groups=((docs["s1"].id,), (docs["s2"].id,))),
        ),
    )
    result = await apply_corrections(session_factory, candidates)

    assert result.splits == [] and result.merges == []
    reasons = {(s.kind, s.reason, s.event_ids) for s in result.skipped}
    assert ("split", SKIP_CAPPED, (capped.id,)) in reasons
    assert ("split", SKIP_CONFLICT, (conflict.id,)) in reasons
    assert ("merge", SKIP_CONFLICT, tuple(sorted((conflict.id, partner.id)))) in reasons
    assert ("split", SKIP_DOCUMENTS_CHANGED, (stale.id,)) in reasons
    async with session_factory() as session:
        assert await _links(session, capped.id) == {docs["c1"].id, docs["c2"].id}
        assert await _links(session, conflict.id) == {docs["k1"].id, docs["k2"].id}
        assert (await session.get(TechEvent, partner.id)).merged_into_id is None


@pytest.mark.asyncio
async def test_second_run_with_same_candidates_changes_nothing(session_factory, clean):
    """같은 후보로 다시 돌면 이미 반영된 병합·분할은 낡은 후보로 건너뛰고 DB는 그대로다.

    야간 재시도는 저장 상태 없이 같은 진입점을 다시 부르는 것이므로, 두 번째
    실행이 링크를 또 옮기거나 사건을 또 만들면 안 된다.
    """
    async with session_factory() as session:
        a = await _event(session, "ia")
        b = await _event(session, "ib")
        sp = await _event(session, "isp")
        da, db = await _doc(session, "ida"), await _doc(session, "idb")
        s1, s2, s3 = [await _doc(session, n) for n in ("is1", "is2", "is3")]
        session.add_all([
            EventDocument(event_id=a.id, tech_item_id=da.id),
            EventDocument(event_id=b.id, tech_item_id=db.id),
            *[EventDocument(event_id=sp.id, tech_item_id=d.id) for d in (s1, s2, s3)],
        ])
        await session.commit()

    candidates = ReclusterCandidates(
        merges=_merges(((a.id, b.id), (da.id, db.id))).merges,
        splits=(SplitCandidate(event_id=sp.id, groups=((s1.id, s2.id), (s3.id,))),),
    )
    first = await apply_corrections(session_factory, candidates)
    assert len(first.merges) == 1 and len(first.splits) == 1

    async def _snapshot():
        async with session_factory() as session:
            events = (
                await session.execute(
                    select(TechEvent.id, TechEvent.merged_into_id).where(
                        TechEvent.id.in_([a.id, b.id, sp.id, *first.splits[0].new_event_ids])
                    )
                )
            ).all()
            links = (
                await session.execute(
                    select(EventDocument.event_id, EventDocument.tech_item_id).where(
                        EventDocument.tech_item_id.in_([da.id, db.id, s1.id, s2.id, s3.id])
                    )
                )
            ).all()
            return sorted(events), sorted(links)

    before = await _snapshot()
    second = await apply_corrections(session_factory, candidates)
    assert second.merges == [] and second.splits == []
    assert [r for r in second.relocations if r.target_id in {da.id, db.id}] == []
    assert any(s.kind == "merge" and s.reason == SKIP_TOMBSTONED for s in second.skipped)
    assert any(
        s.kind == "split" and s.reason == SKIP_DOCUMENTS_CHANGED and s.event_ids == (sp.id,)
        for s in second.skipped
    )
    assert await _snapshot() == before


@pytest.mark.asyncio
async def test_split_with_overlapping_or_single_fragment_is_rejected(session_factory, clean):
    """조각이 파티션이 아니면(겹침·한 조각) 가르지 않고 사유를 남긴다."""
    async with session_factory() as session:
        ov = await _event(session, "overlap")
        one = await _event(session, "single")
        o1, o2, o3, x1 = [await _doc(session, n) for n in ("o1", "o2", "o3", "x1")]
        session.add_all([
            *[EventDocument(event_id=ov.id, tech_item_id=d.id) for d in (o1, o2, o3)],
            EventDocument(event_id=one.id, tech_item_id=x1.id),
        ])
        await session.commit()

    result = await apply_corrections(
        session_factory,
        ReclusterCandidates(
            merges=(),
            splits=(
                SplitCandidate(event_id=ov.id, groups=((o1.id, o2.id), (o2.id, o3.id))),
                SplitCandidate(event_id=one.id, groups=((x1.id,), ())),
            ),
        ),
    )

    assert result.splits == []
    reasons = {(s.reason, s.event_ids) for s in result.skipped if s.kind == "split"}
    assert (SKIP_INVALID_FRAGMENTS, (ov.id,)) in reasons
    assert (SKIP_INVALID_FRAGMENTS, (one.id,)) in reasons
    async with session_factory() as session:
        assert await _links(session, ov.id) == {o1.id, o2.id, o3.id}
