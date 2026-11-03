"""사건 상세의 '비슷한 기술의 최근 소식' 라우트 테스트 (ARG-329). DB 없음."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.web.app import _get_session, build_web_app
from argos.web.services.event_detail import EventDetailView, EvidenceDoc
from argos.web.services.similar import SimilarNews
from argos.web.services.technology import BranchRef, TechChip, TechEventRef

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class _FakeSession:
    def __init__(self, rollback_fails: bool = False) -> None:
        self.added: list = []
        self.commits = 0
        self.rollbacks = 0
        self._rollback_fails = rollback_fails

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1
        if self._rollback_fails:
            raise RuntimeError("rollback boom")


def _view(rep_id=None) -> EventDetailView:
    return EventDetailView(
        id=uuid.uuid4(), title="사건", summary="요약",
        documents=[
            EvidenceDoc(id=uuid.uuid4(), title="근거 기사", source_url="https://a.example/x",
                        domain="a.example", image_url=None, summary=None,
                        reported_at=T0, is_first=True),
        ],
        rep_id=rep_id, category=CategoryType.MAINSTREAM, trust_score=None, image_url=None,
    )


def _news(name: str = "비슷한기술") -> SimilarNews:
    return SimilarNews(
        entity_id=uuid.uuid4(), name=name,
        branch=BranchRef(id=uuid.uuid4(), name="코딩 에이전트"),
        event=TechEventRef(id=uuid.uuid4(), title="그쪽 사건 제목", occurred_at=T0,
                           event_type_label=None),
    )


def _chips() -> list[TechChip]:
    return [TechChip(entity_id=uuid.uuid4(), name="칩기술", doc_count=2)]


def _client(monkeypatch, view, *, chips=None, news=None, news_fn=None, chips_fn=None,
            session=None):
    session = session or _FakeSession()
    app = build_web_app()

    async def _fake_session():
        yield session

    app.dependency_overrides[_get_session] = _fake_session

    async def _resolve(s, event_id):
        return event_id

    async def _fetch(s, event_id):
        return view

    async def _default_chips(s, event_ids):
        return {view.id: chips} if chips else {}

    async def _default_news(s, event_id, chip_ids):
        return tuple(news or ())

    async def _no_ctx(s, item_id):
        return None

    monkeypatch.setattr("argos.web.app.resolve_event", _resolve)
    monkeypatch.setattr("argos.web.app.fetch_event_detail", _fetch)
    monkeypatch.setattr("argos.web.app.fetch_event_tech_chips", chips_fn or _default_chips)
    monkeypatch.setattr("argos.web.app.fetch_event_similar_news", news_fn or _default_news)
    monkeypatch.setattr("argos.web.app._load_feed_card_context", _no_ctx)
    return TestClient(app, raise_server_exceptions=False), session


def test_news_rendered_below_evidence(monkeypatch):
    view, n = _view(), _news()
    client, _ = _client(monkeypatch, view, chips=_chips(), news=[n])
    body = client.get(f"/event/{view.id}").text
    assert "비슷한 기술의 최근 소식" in body
    assert body.index("근거 기사") < body.index("비슷한 기술의 최근 소식") < body.index("등장한 기술")
    assert f'href="/event/{n.event.id}"' in body
    assert f'href="/tech/{n.entity_id}"' in body
    assert f'href="/branch/{n.branch.id}"' in body


def test_news_shows_branch_not_colisted(monkeypatch):
    view = _view()
    client, _ = _client(monkeypatch, view, chips=_chips(), news=[_news()])
    body = client.get(f"/event/{view.id}").text
    assert "코딩 에이전트" in body
    assert "함께 나열됨" not in body


def test_area_hidden_when_empty(monkeypatch):
    view = _view()
    client, _ = _client(monkeypatch, view, chips=_chips(), news=[])
    body = client.get(f"/event/{view.id}").text
    assert "비슷한 기술의 최근 소식" not in body
    assert 'class="similar-news"' not in body


def _boom(*a, **k):
    raise RuntimeError("boom")


async def _raise(s, event_id, chip_ids):
    raise RuntimeError("boom")


def test_failure_hides_area_keeps_click(monkeypatch):
    from argos.models.feed_event import FeedEvent

    rep = uuid.uuid4()
    view = _view(rep_id=rep)
    client, session = _client(monkeypatch, view, chips=_chips(), news_fn=_raise)
    resp = client.get(f"/event/{view.id}")
    assert resp.status_code == 200
    assert "비슷한 기술의 최근 소식" not in resp.text
    assert "근거 기사" in resp.text
    assert any(isinstance(o, FeedEvent) for o in session.added)
    assert session.commits == 1
    assert session.rollbacks == 1


def test_failure_with_rollback_failure_still_renders(monkeypatch):
    rep = uuid.uuid4()
    view = _view(rep_id=rep)
    session = _FakeSession(rollback_fails=True)
    client, session = _client(monkeypatch, view, chips=_chips(), news_fn=_raise, session=session)
    resp = client.get(f"/event/{view.id}")
    assert resp.status_code == 200
    assert "비슷한 기술의 최근 소식" not in resp.text
    assert session.commits == 1


def test_not_called_when_chip_query_failed(monkeypatch):
    view, calls = _view(), []

    async def _chips_boom(s, event_ids):
        raise RuntimeError("chips boom")

    async def _news(s, event_id, chip_ids):
        calls.append(chip_ids)
        return ()

    client, _ = _client(monkeypatch, view, chips_fn=_chips_boom, news_fn=_news)
    assert client.get(f"/event/{view.id}").status_code == 200
    assert calls == []


def test_chip_ids_passed_to_lookup(monkeypatch):
    view, calls, chips = _view(), [], _chips()

    async def _news(s, event_id, chip_ids):
        calls.append((event_id, list(chip_ids)))
        return ()

    client, _ = _client(monkeypatch, view, chips=chips, news_fn=_news)
    client.get(f"/event/{view.id}")
    assert calls == [(view.id, [chips[0].entity_id])]
