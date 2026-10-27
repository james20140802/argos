"""Route + template tests for the 기술 페이지 (ARG-318). No live database."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.technology import BranchRef, TechEventRef, TechPageView

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
CSS = Path(__file__).resolve().parents[2] / "src/argos/web/static/css/argos.css"


class _FakeSession:
    async def commit(self) -> None:
        return None


def _client(monkeypatch, *, resolve_to="same", view=None) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session

    async def _resolve(session, entity_id):
        return entity_id if resolve_to == "same" else resolve_to

    async def _fetch(session, canonical_id):
        return view

    monkeypatch.setattr("argos.web.app.resolve_canonical_id", _resolve)
    monkeypatch.setattr("argos.web.app.fetch_tech_page", _fetch)
    return TestClient(app, raise_server_exceptions=False)


def _view(**kw) -> TechPageView:
    defaults = dict(
        entity_id=uuid.uuid4(),
        name="Claude Duet",
        description="에이전트형 개인 비서",
        branch_path=(
            BranchRef(id=uuid.uuid4(), name="AI 에이전트 및 어시스턴트"),
            BranchRef(id=uuid.uuid4(), name="개인·업무 비서형 에이전트"),
        ),
        aliases=("Duet",),
        events=(),
    )
    defaults.update(kw)
    return TechPageView(**defaults)


def test_tech_page_renders_all_parts(monkeypatch):
    a, b = uuid.uuid4(), uuid.uuid4()
    view = _view(
        events=(
            TechEventRef(id=b, title="사건 B", occurred_at=T0, event_type_label="출시"),
            TechEventRef(id=a, title="사건 A", occurred_at=T0, event_type_label=None),
        )
    )
    resp = _client(monkeypatch, view=view).get(f"/tech/{view.entity_id}")
    assert resp.status_code == 200
    body = resp.text
    for needle in ("Claude Duet", "에이전트형 개인 비서", "AI 에이전트 및 어시스턴트",
                   " › ", "개인·업무 비서형 에이전트", "다른 이름", "Duet", "2026-10-01"):
        assert needle in body
    assert body.index(f'href="/event/{b}"') < body.index(f'href="/event/{a}"')


def test_tech_page_empty_timeline_message(monkeypatch):
    view = _view()
    body = _client(monkeypatch, view=view).get(f"/tech/{view.entity_id}").text
    assert "아직 이 기술이 등장한 사건이 없습니다." in body


def test_tech_page_unclassified_label(monkeypatch):
    view = _view(branch_path=())
    body = _client(monkeypatch, view=view).get(f"/tech/{view.entity_id}").text
    assert "미분류" in body


def test_tech_alias_redirects_to_canonical(monkeypatch):
    canonical = uuid.uuid4()
    resp = _client(monkeypatch, resolve_to=canonical, view=_view()).get(
        f"/tech/{uuid.uuid4()}", follow_redirects=False
    )
    assert resp.status_code == 307
    assert resp.headers["location"] == f"/tech/{canonical}"


def test_tech_not_technology_is_404(monkeypatch):
    assert _client(monkeypatch, view=None).get(f"/tech/{uuid.uuid4()}").status_code == 404


def test_tech_unknown_id_is_404(monkeypatch):
    assert _client(monkeypatch, resolve_to=None, view=_view()).get(
        f"/tech/{uuid.uuid4()}"
    ).status_code == 404


def test_tech_bad_uuid_is_404(monkeypatch):
    assert _client(monkeypatch, view=_view()).get("/tech/not-a-uuid").status_code == 404


def test_tech_page_without_description_has_no_empty_paragraph(monkeypatch):
    view = _view(description=None)
    body = _client(monkeypatch, view=view).get(f"/tech/{view.entity_id}").text
    assert "tech-description" not in body


def test_tech_page_mobile_css():
    css = CSS.read_text()
    assert ".tech-" in css
    for selector in (".tech-path", ".tech-timeline"):
        block = css[css.index(selector):]
        block = block[: block.index("}")]
        assert "flex-wrap: wrap" in block, selector
    for selector in (".tech-name", ".tech-path__branch", ".tech-timeline__title"):
        block = css[css.index(selector):]
        block = block[: block.index("}")]
        assert "overflow-wrap: anywhere" in block, selector


def test_merged_event_link_still_redirects(monkeypatch):
    old, survivor = uuid.uuid4(), uuid.uuid4()

    async def _resolve(session, event_id):
        return survivor

    client = _client(monkeypatch, view=None)
    monkeypatch.setattr("argos.web.app.resolve_event", _resolve)
    resp = client.get(f"/event/{old}", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == f"/event/{survivor}"
