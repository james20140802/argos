"""Route + template tests for the 갈래 페이지 (ARG-320). No live database."""
from __future__ import annotations

import uuid
from pathlib import Path

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.technology import (
    BranchGroup,
    BranchPageView,
    BranchRef,
    TechPageView,
    TechRef,
)

CSS = Path(__file__).resolve().parents[2] / "src/argos/web/static/css/argos.css"


class _FakeSession:
    async def commit(self) -> None:
        return None


def _client(monkeypatch, *, branch=None, tech=None) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session

    async def _fetch_branch(session, branch_id):
        return branch

    async def _resolve(session, entity_id):
        return entity_id

    async def _fetch_tech(session, canonical_id):
        return tech

    monkeypatch.setattr("argos.web.app.fetch_branch_page", _fetch_branch)
    monkeypatch.setattr("argos.web.app.resolve_canonical_id", _resolve)
    monkeypatch.setattr("argos.web.app.fetch_tech_page", _fetch_tech)
    return TestClient(app, raise_server_exceptions=False)


def _leaf_view():
    top = BranchRef(id=uuid.uuid4(), name="위 갈래")
    leaf = BranchRef(id=uuid.uuid4(), name="아래 갈래")
    t1, t2 = TechRef(uuid.uuid4(), "Alpha"), TechRef(uuid.uuid4(), "Zed")
    view = BranchPageView(
        branch=leaf, parent=top, description="아래 설명",
        groups=(BranchGroup(branch=leaf, techs=(t1, t2)),),
    )
    return view, top, t1, t2


def test_branch_page_lists_techs_linking_to_tech_page(monkeypatch):
    view, _top, t1, t2 = _leaf_view()
    body = _client(monkeypatch, branch=view).get(f"/branch/{view.branch.id}").text
    assert "아래 설명" in body
    assert body.index(f'href="/tech/{t1.entity_id}"') < body.index(f'href="/tech/{t2.entity_id}"')


def test_top_branch_page_shows_child_groups(monkeypatch):
    top = BranchRef(id=uuid.uuid4(), name="위 갈래")
    c1, c2 = BranchRef(uuid.uuid4(), "첫째"), BranchRef(uuid.uuid4(), "둘째")
    t = TechRef(uuid.uuid4(), "Alpha")
    view = BranchPageView(
        branch=top, parent=None, description="설명",
        groups=(BranchGroup(c1, (t,)), BranchGroup(c2, ())),
    )
    body = _client(monkeypatch, branch=view).get(f"/branch/{top.id}").text
    assert f'href="/branch/{c1.id}"' in body and f'href="/branch/{c2.id}"' in body
    assert f'href="/tech/{t.entity_id}"' in body
    assert "아직 이 갈래에 배정된 기술이 없습니다." in body


def test_leaf_branch_shows_parent_link(monkeypatch):
    view, top, _, _ = _leaf_view()
    body = _client(monkeypatch, branch=view).get(f"/branch/{view.branch.id}").text
    assert f'href="/branch/{top.id}"' in body and " › " in body


def test_branch_unknown_is_404(monkeypatch):
    assert _client(monkeypatch, branch=None).get(f"/branch/{uuid.uuid4()}").status_code == 404


def test_branch_bad_uuid_is_404(monkeypatch):
    assert _client(monkeypatch).get("/branch/not-a-uuid").status_code == 404


def _tech(path):
    return TechPageView(
        entity_id=uuid.uuid4(), name="Claude Duet", description=None,
        branch_path=path, aliases=(), events=(),
    )


def test_tech_page_branch_path_links(monkeypatch):
    path = (BranchRef(uuid.uuid4(), "위"), BranchRef(uuid.uuid4(), "아래"))
    view = _tech(path)
    body = _client(monkeypatch, tech=view).get(f"/tech/{view.entity_id}").text
    for b in path:
        assert f'href="/branch/{b.id}"' in body


def test_tech_page_unclassified_not_link(monkeypatch):
    view = _tech(())
    body = _client(monkeypatch, tech=view).get(f"/tech/{view.entity_id}").text
    assert "미분류" in body
    before = body[: body.index("미분류")]
    assert before.count("<a ") == before.count("</a>")  # 열린 <a> 안이 아니다


def test_branch_mobile_css():
    css = CSS.read_text()
    for selector in (".branch-groups", ".branch-techs"):
        block = css[css.index(selector):]
        block = block[: block.index("}")]
        assert "flex-wrap: wrap" in block, selector
    for selector in (".branch-description", ".branch-group__name", ".branch-techs li"):
        block = css[css.index(selector):]
        block = block[: block.index("}")]
        assert "overflow-wrap: anywhere" in block, selector


def test_top_branch_without_children_shows_empty_message(monkeypatch):
    top = BranchRef(id=uuid.uuid4(), name="위 갈래")
    view = BranchPageView(branch=top, parent=None, description="설명", groups=())
    resp = _client(monkeypatch, branch=view).get(f"/branch/{top.id}")
    assert resp.status_code == 200
    assert "아직 이 갈래에 속한 기술이 없습니다." in resp.text
