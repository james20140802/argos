"""ARG-327: 기술 페이지의 '비슷한 역할' 영역 — No live database."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.similar import SimilarTech
from argos.web.services.technology import BranchRef, TechEventRef, TechPageView

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class _FakeSession:
    def __init__(self, rollback_raises: bool = False) -> None:
        self._rollback_raises = rollback_raises

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        if self._rollback_raises:
            raise RuntimeError("rollback failed")


def _view(**kw) -> TechPageView:
    defaults = dict(
        entity_id=uuid.uuid4(),
        name="Claude Duet",
        description="에이전트형 개인 비서",
        branch_path=(BranchRef(id=uuid.uuid4(), name="위"), BranchRef(id=uuid.uuid4(), name="아래")),
        aliases=(),
        events=(TechEventRef(id=uuid.uuid4(), title="타임라인 사건", occurred_at=T0, event_type_label=None),),
    )
    defaults.update(kw)
    return TechPageView(**defaults)


def _similar(**kw) -> SimilarTech:
    defaults = dict(
        entity_id=uuid.uuid4(),
        name="Cursor",
        branch=BranchRef(id=uuid.uuid4(), name="코딩 에이전트"),
        latest_event=TechEventRef(
            id=uuid.uuid4(), title="Cursor 3 출시", occurred_at=T0, event_type_label="출시"
        ),
        colisted_count=0,
        colisted_example=None,
    )
    defaults.update(kw)
    return SimilarTech(**defaults)


def _client(monkeypatch, *, similar=(), fetch_raises=False, rollback_raises=False) -> tuple[TestClient, TechPageView]:
    app = build_web_app()
    view = _view()

    async def _fake_session():
        yield _FakeSession(rollback_raises)

    app.dependency_overrides[_get_session] = _fake_session

    async def _resolve(session, entity_id):
        return entity_id

    async def _fetch(session, canonical_id):
        return view

    async def _fetch_similar(session, canonical_id):
        if fetch_raises:
            raise RuntimeError("boom")
        return similar

    monkeypatch.setattr("argos.web.app.resolve_canonical_id", _resolve)
    monkeypatch.setattr("argos.web.app.fetch_tech_page", _fetch)
    monkeypatch.setattr("argos.web.app.fetch_similar_techs", _fetch_similar)
    return TestClient(app, raise_server_exceptions=False), view


def test_area_shows_names_events_branch_and_colisted(monkeypatch):
    s = _similar(colisted_count=4, colisted_example="Cursor와 Windsurf 비교")
    client, view = _client(monkeypatch, similar=(s,))
    body = client.get(f"/tech/{view.entity_id}").text
    assert "비슷한 역할" in body
    assert "Cursor" in body and f'href="/tech/{s.entity_id}"' in body
    assert "Cursor 3 출시" in body and f'href="/event/{s.latest_event.id}"' in body
    assert "코딩 에이전트" in body and f'href="/branch/{s.branch.id}"' in body
    assert "기사에서 함께 나열됨" in body and "4회" in body
    assert "Cursor와 Windsurf 비교" in body


def test_no_colisted_text_without_evidence(monkeypatch):
    client, view = _client(monkeypatch, similar=(_similar(),))
    assert "기사에서 함께 나열됨" not in client.get(f"/tech/{view.entity_id}").text


def test_event_title_absent_when_no_event(monkeypatch):
    s = _similar(latest_event=None)
    client, view = _client(monkeypatch, similar=(s,))
    body = client.get(f"/tech/{view.entity_id}").text
    assert "Cursor" in body and "Cursor 3 출시" not in body


def test_area_hidden_when_empty(monkeypatch):
    client, view = _client(monkeypatch, similar=())
    resp = client.get(f"/tech/{view.entity_id}")
    assert resp.status_code == 200 and "비슷한 역할" not in resp.text


def test_area_hidden_when_query_fails(monkeypatch):
    client, view = _client(monkeypatch, fetch_raises=True)
    resp = client.get(f"/tech/{view.entity_id}")
    assert resp.status_code == 200
    assert "비슷한 역할" not in resp.text and "타임라인 사건" in resp.text


def test_page_survives_rollback_failure(monkeypatch):
    client, view = _client(monkeypatch, fetch_raises=True, rollback_raises=True)
    resp = client.get(f"/tech/{view.entity_id}")
    assert resp.status_code == 200 and "타임라인 사건" in resp.text


def test_area_is_between_header_and_timeline(monkeypatch):
    client, view = _client(monkeypatch, similar=(_similar(),))
    body = client.get(f"/tech/{view.entity_id}").text
    assert body.index("</header>") < body.index("비슷한 역할") < body.index("사건 타임라인")
