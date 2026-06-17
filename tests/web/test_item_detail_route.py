"""Route + template tests for the 상세 보기 screen (ARG-158).

These exercise GET /item/{id} without a live database: the per-request
session dependency is overridden and ``fetch_item_detail`` is monkeypatched
to return a canned ``ItemDetailView`` (or ``None`` for the 404 path). This
keeps the tests runnable on release.yml CI (no Postgres).
"""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import ItemDetailView


def _view(
    *,
    title: str = "GPT-5 launches with multimodal reasoning",
    image_url: str | None = None,
    summary: str | None = "It's the next milestone.",
    category: CategoryType | None = CategoryType.MAINSTREAM,
    trust_score: float | None = 0.82,
    source_url: str = "https://example.com/gpt5",
) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title=title,
        source_url=source_url,
        image_url=image_url,
        summary=summary,
        category=category,
        trust_score=trust_score,
        published_at=None,
    )


def _client_with_detail(
    monkeypatch, view: ItemDetailView | None, capture: list | None = None
) -> TestClient:
    """Build a TestClient whose detail route returns ``view`` without DB access."""
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_item_detail(session, item_id):
        if capture is not None:
            capture.append({"item_id": item_id})
        return view

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch_item_detail)
    return TestClient(app, raise_server_exceptions=False)


def test_item_detail_renders_title_summary_and_source_link(monkeypatch):
    view = _view(
        title="Claude Opus 4.8 ships",
        summary="A leap in coding reliability.",
        source_url="https://example.com/opus-4-8",
    )
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "Claude Opus 4.8 ships" in body
    assert "A leap in coding reliability." in body
    assert "https://example.com/opus-4-8" in body
    # The 원문 보기 link must open in a new tab with safe rel attributes.
    assert 'target="_blank"' in body
    assert "noopener" in body
    assert "noreferrer" in body
    assert "원문 보기" in body


def test_item_detail_renders_hero_image_when_url_present(monkeypatch):
    view = _view(image_url="https://cdn.example.com/hero.jpg")
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "https://cdn.example.com/hero.jpg" in body
    assert "<img" in body
    assert "detail-hero__img" in body
    # Fallback marker must NOT appear when we have an image.
    assert "detail-hero--fallback" not in body


def test_item_detail_renders_category_tinted_fallback_when_no_image(monkeypatch):
    view = _view(image_url=None, category=CategoryType.ALPHA)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "detail-hero--fallback" in body
    assert "detail-hero--alpha" in body
    assert "<img" not in body or "detail-hero__img" not in body


def test_item_detail_renders_trust_score_dial(monkeypatch):
    view = _view(trust_score=0.73)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "trust-dial" in body
    # 0.73 → 73% on the conic dial.
    assert "73" in body
    assert "신뢰도" in body


def test_item_detail_omits_trust_dial_when_score_is_none(monkeypatch):
    view = _view(trust_score=None)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "trust-dial" not in resp.text


def test_item_detail_returns_404_for_unknown_id(monkeypatch):
    capture: list = []
    client = _client_with_detail(monkeypatch, view=None, capture=capture)

    item_id = uuid.uuid4()
    resp = client.get(f"/item/{item_id}")

    assert resp.status_code == 404
    body = resp.text
    assert "404" in body
    assert "관측 대상이 없습니다" in body
    # Route must call the service exactly once with the parsed UUID.
    assert capture == [{"item_id": item_id}]


def test_item_detail_returns_404_for_malformed_uuid(monkeypatch):
    capture: list = []
    client = _client_with_detail(monkeypatch, view=None, capture=capture)

    resp = client.get("/item/not-a-uuid")

    assert resp.status_code == 404
    assert "관측 대상이 없습니다" in resp.text
    # Service must never be called with a non-UUID.
    assert capture == []


def test_item_detail_inherits_base_layout(monkeypatch):
    view = _view()
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    # Masthead + tabbar from base.html must be present.
    assert "ARGOS" in body
    assert "관측 피드" in body
    assert "포트폴리오" in body
