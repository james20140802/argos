"""ARG-328: 기술 페이지 타임라인의 '비슷한 기술 포함' 토글 — No live database."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.similar import MixedEventRef, SimilarTech
from argos.web.services.technology import BranchRef, TechEventRef, TechPageView, TechRef

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class _FakeSession:
    def __init__(self) -> None:
        self.rolled_back = False

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        self.rolled_back = True


def _view() -> TechPageView:
    return TechPageView(
        entity_id=uuid.uuid4(),
        name="Claude Duet",
        description=None,
        branch_path=(BranchRef(id=uuid.uuid4(), name="위"), BranchRef(id=uuid.uuid4(), name="아래")),
        aliases=(),
        events=(TechEventRef(id=uuid.uuid4(), title="내 사건 제목", occurred_at=T0, event_type_label=None),),
    )


def _similar() -> SimilarTech:
    return SimilarTech(
        entity_id=uuid.uuid4(),
        name="Cursor",
        branch=BranchRef(id=uuid.uuid4(), name="코딩 에이전트"),
        latest_event=None,
    )


class _Ctx:
    pass


def _client(monkeypatch, *, similar=None, mixed_raises=False, found=True, canonical=None):
    app = build_web_app()
    ctx = _Ctx()
    ctx.view = _view()
    ctx.similar = (_similar(),) if similar is None else similar
    ctx.session = _FakeSession()
    ctx.mixed_args = []
    techs = [TechRef(entity_id=ctx.view.entity_id, name="Claude Duet")]
    techs += [TechRef(entity_id=t.entity_id, name=t.name) for t in ctx.similar[:1]]
    ctx.mixed = (
        MixedEventRef(
            id=uuid.uuid4(),
            title="섞인 사건 제목",
            occurred_at=T0,
            event_type_label="출시",
            techs=tuple(techs),
        ),
    )

    async def _fake_session():
        yield ctx.session

    app.dependency_overrides[_get_session] = _fake_session

    async def _resolve(session, entity_id):
        return canonical if canonical is not None else entity_id

    async def _fetch(session, canonical_id):
        return ctx.view if found else None

    async def _fetch_similar(session, canonical_id):
        return ctx.similar

    async def _fetch_mixed(session, own, similar):
        ctx.mixed_args.append((own, similar))
        if mixed_raises:
            raise RuntimeError("boom")
        return ctx.mixed

    monkeypatch.setattr("argos.web.app.resolve_canonical_id", _resolve)
    monkeypatch.setattr("argos.web.app.fetch_tech_page", _fetch)
    monkeypatch.setattr("argos.web.app.fetch_similar_techs", _fetch_similar)
    monkeypatch.setattr("argos.web.app.fetch_mixed_timeline", _fetch_mixed)
    return TestClient(app, raise_server_exceptions=False, follow_redirects=False), ctx


def test_toggle_off_matches_existing_timeline(monkeypatch):
    client, ctx = _client(monkeypatch)
    body = client.get(f"/tech/{ctx.view.entity_id}").text
    assert "내 사건 제목" in body
    assert "섞인 사건 제목" not in body
    assert "tech-timeline__techs" not in body
    assert 'class="tech-events"' in body and 'aria-label="사건 타임라인"' in body
    assert ctx.mixed_args == []


def test_toggle_off_empty_state_text_unchanged(monkeypatch):
    client, ctx = _client(monkeypatch)
    ctx.view = TechPageView(**{**ctx.view.__dict__, "events": ()})
    body = client.get(f"/tech/{ctx.view.entity_id}").text
    assert "아직 이 기술이 등장한 사건이 없습니다." in body


def test_toggle_on_empty_state_mentions_similar_techs(monkeypatch):
    client, ctx = _client(monkeypatch)
    ctx.mixed = ()
    body = client.get(f"/tech/{ctx.view.entity_id}?similar=1").text
    assert "아직 이 기술과 비슷한 기술이 등장한 사건이 없습니다." in body


def test_toggle_hidden_without_similar(monkeypatch):
    client, ctx = _client(monkeypatch, similar=())
    body = client.get(f"/tech/{ctx.view.entity_id}?similar=1").text
    assert "비슷한 기술 포함" not in body
    assert "내 사건 제목" in body
    assert ctx.mixed_args == []


def test_similar_param_turns_on_mix(monkeypatch):
    client, ctx = _client(monkeypatch)
    body = client.get(f"/tech/{ctx.view.entity_id}?similar=1").text
    assert "섞인 사건 제목" in body
    assert "내 사건 제목" not in body
    assert f'href="/tech/{ctx.similar[0].entity_id}"' in body.split('id="tech-timeline"')[1]
    tag = body.split('id="tech-toggle"')[1].split(">")[0]
    assert 'aria-current="true"' in tag and "role=" not in tag and "aria-checked" not in body


def test_invalid_param_is_off(monkeypatch):
    client, ctx = _client(monkeypatch)
    for value in ("0", "true", "", "2", "1 "):
        body = client.get(f"/tech/{ctx.view.entity_id}", params={"similar": value}).text
        assert "내 사건 제목" in body and "섞인 사건 제목" not in body
        assert 'id="tech-toggle"' in body and 'aria-current' not in body.split('id="tech-toggle"')[1].split(">")[0]
    assert ctx.mixed_args == []


def test_toggle_links_point_to_next_state(monkeypatch):
    client, ctx = _client(monkeypatch)
    tid = ctx.view.entity_id
    off = client.get(f"/tech/{tid}").text
    assert f'href="/tech/{tid}?similar=1"' in off
    assert f'hx-get="/tech/{tid}/timeline?similar=1"' in off
    assert f'hx-push-url="/tech/{tid}?similar=1"' in off
    on = client.get(f"/tech/{tid}?similar=1").text
    assert f'href="/tech/{tid}"' in on.split('class="tech-toggle"')[1].split(">")[0]
    assert f'hx-get="/tech/{tid}/timeline"' in on
    assert f'hx-push-url="/tech/{tid}"' in on
    assert 'hx-target="#tech-timeline"' in on and 'hx-swap="outerHTML"' in on


def test_partial_route_returns_fragment_only(monkeypatch):
    client, ctx = _client(monkeypatch)
    resp = client.get(f"/tech/{ctx.view.entity_id}/timeline?similar=1")
    assert resp.status_code == 200
    assert "<html" not in resp.text
    assert 'id="tech-timeline"' in resp.text
    assert "섞인 사건 제목" in resp.text


def test_partial_route_not_found(monkeypatch):
    client, ctx = _client(monkeypatch, found=False)
    bad = client.get("/tech/not-a-uuid/timeline")
    missing = client.get(f"/tech/{ctx.view.entity_id}/timeline")
    for resp in (bad, missing):
        assert resp.status_code == 404
        assert 'id="tech-timeline"' not in resp.text


def test_alias_redirect_keeps_query(monkeypatch):
    canon = uuid.uuid4()
    client, ctx = _client(monkeypatch, canonical=canon)
    alias = uuid.uuid4()
    full = client.get(f"/tech/{alias}?similar=1")
    assert full.status_code == 307
    assert full.headers["location"] == f"/tech/{canon}?similar=1"
    assert client.get(f"/tech/{alias}").headers["location"] == f"/tech/{canon}"
    part = client.get(f"/tech/{alias}/timeline?similar=1")
    assert part.status_code == 307
    assert part.headers["location"] == f"/tech/{canon}/timeline?similar=1"


def test_mixed_query_failure_falls_back_to_own(monkeypatch):
    client, ctx = _client(monkeypatch, mixed_raises=True)
    resp = client.get(f"/tech/{ctx.view.entity_id}?similar=1")
    assert resp.status_code == 200
    assert "내 사건 제목" in resp.text and "섞인 사건 제목" not in resp.text
    assert 'id="tech-toggle"' in resp.text
    assert "aria-current" not in resp.text.split('id="tech-toggle"')[1].split(">")[0]
    assert ctx.session.rolled_back


def test_mixed_uses_same_similar_list(monkeypatch):
    client, ctx = _client(monkeypatch)
    body = client.get(f"/tech/{ctx.view.entity_id}?similar=1").text
    own, similar = ctx.mixed_args[0]
    assert similar == ctx.similar
    assert own == TechRef(entity_id=ctx.view.entity_id, name=ctx.view.name)
    assert "비슷한 역할" in body
