"""ARG-329: 사건 상세의 '비슷한 기술의 최근 소식' 합치기 — DB 없이 돈다.

칩별 조회(``_similar``)를 바꿔 끼우므로 세션은 쓰이지 않는다.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from argos.web.services import similar as similar_mod
from argos.web.services.similar import SIMILAR_NEWS_LIMIT, SimilarTech, fetch_event_similar_news
from argos.web.services.technology import BranchRef, TechEventRef

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
BRANCH = BranchRef(id=uuid.uuid4(), name="코딩 에이전트")


def _ev(days: int = 0, eid: uuid.UUID | None = None) -> TechEventRef:
    return TechEventRef(
        id=eid or uuid.uuid4(), title="사건", occurred_at=T0 + timedelta(days=days),
        event_type_label=None,
    )


def _tech(name: str = "T", event: TechEventRef | None = None, eid=None) -> SimilarTech:
    return SimilarTech(entity_id=eid or uuid.uuid4(), name=name, branch=BRANCH, latest_event=event)


def _patch(monkeypatch, mapping: dict, calls: list | None = None):
    async def _fake(lookup, canonical_id):
        if calls is not None:
            calls.append(canonical_id)
        return tuple(mapping.get(canonical_id, ()))

    async def _noop(self, ids):
        return None

    monkeypatch.setattr(similar_mod, "_similar", _fake)
    monkeypatch.setattr(similar_mod._Lookup, "load_assignments", _noop)
    monkeypatch.setattr(similar_mod._Lookup, "load_bundles", _noop)
    monkeypatch.setattr(similar_mod._Lookup, "load_events", _noop)


async def test_union_across_chips_counts_once(monkeypatch):
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    shared = _tech("공통", _ev(1))
    _patch(monkeypatch, {c1: [shared], c2: [shared, _tech("다른", _ev(2))]})
    out = await fetch_event_similar_news(None, uuid.uuid4(), [c1, c2])
    assert [n.entity_id for n in out].count(shared.entity_id) == 1
    assert len(out) == 2


async def test_chip_techs_excluded(monkeypatch):
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    _patch(monkeypatch, {c1: [_tech("칩2", _ev(1), eid=c2), _tech("남", _ev(2))]})
    out = await fetch_event_similar_news(None, uuid.uuid4(), [c1, c2])
    assert [n.name for n in out] == ["남"]


async def test_this_event_excluded(monkeypatch):
    c1, this = uuid.uuid4(), uuid.uuid4()
    _patch(monkeypatch, {c1: [_tech("자기", _ev(5, eid=this)), _tech("남", _ev(1))]})
    out = await fetch_event_similar_news(None, this, [c1])
    assert [n.name for n in out] == ["남"]


async def test_same_event_shown_once(monkeypatch):
    c1, shared_event = uuid.uuid4(), uuid.uuid4()
    _patch(monkeypatch, {c1: [_tech("A", _ev(1, eid=shared_event)), _tech("B", _ev(1, eid=shared_event))]})
    out = await fetch_event_similar_news(None, uuid.uuid4(), [c1])
    assert len(out) == 1


async def test_limit_three_most_recent(monkeypatch):
    c1 = uuid.uuid4()
    techs = [_tech(f"T{d}", _ev(d)) for d in (1, 5, 3, 4, 2)]
    _patch(monkeypatch, {c1: techs})
    out = await fetch_event_similar_news(None, uuid.uuid4(), [c1])
    assert SIMILAR_NEWS_LIMIT == 3
    assert [n.name for n in out] == ["T5", "T4", "T3"]


async def test_tech_without_event_excluded(monkeypatch):
    c1 = uuid.uuid4()
    _patch(monkeypatch, {c1: [_tech("없음", None), _tech("있음", _ev(1))]})
    out = await fetch_event_similar_news(None, uuid.uuid4(), [c1])
    assert [n.name for n in out] == ["있음"]


async def test_no_chips_no_query(monkeypatch):
    calls: list = []
    _patch(monkeypatch, {}, calls)
    assert await fetch_event_similar_news(None, uuid.uuid4(), []) == ()
    assert calls == []


async def test_news_carries_href_branch_event(monkeypatch):
    c1 = uuid.uuid4()
    t = _tech("A", _ev(1))
    _patch(monkeypatch, {c1: [t]})
    (n,) = await fetch_event_similar_news(None, uuid.uuid4(), [c1])
    assert n.href == f"/tech/{t.entity_id}"
    assert n.branch == BRANCH and n.event == t.latest_event
