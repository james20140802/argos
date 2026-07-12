"""Route tests for 🔭 pgvector similarity subsection on /item/{id} (ARG-160)."""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import ItemDetailView, SimilarItem


def _view_with(similar: list[SimilarItem] | None = None) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title="Anchor",
        source_url="https://example.com/anchor",
        image_url=None,
        summary=None,
        category=None,
        trust_score=None,
        published_at=None,
        similar=similar or [],
    )


def _client(monkeypatch, view: ItemDetailView) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_item_detail(session, item_id):
        return view

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch_item_detail)
    return TestClient(app)


def test_related_signals_section_omitted_when_no_similar(monkeypatch):
    view = _view_with(similar=[])
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "관련 신호" not in body
    # The OOB wrapper div (id="detail-signals-<id>") is always present as a swap
    # target; it's the inner <section class="detail-signals"> that must be
    # omitted when there's no subsection content.
    assert 'class="detail-signals"' not in body
    assert "signals-similar" not in body


def test_similarity_subsection_renders_titles_and_links(monkeypatch):
    s1_id, s2_id = uuid.uuid4(), uuid.uuid4()
    view = _view_with(
        similar=[
            SimilarItem(tech_id=s1_id, title="Mistral-Embed v2"),
            SimilarItem(tech_id=s2_id, title="Voyage-3 multilingual"),
        ]
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "관련 신호" in body
    assert "유사 신호" in body
    assert "Mistral-Embed v2" in body
    assert "Voyage-3 multilingual" in body
    assert f"/item/{s1_id}" in body
    assert f"/item/{s2_id}" in body


def test_similar_item_dataclass_is_frozen():
    s = SimilarItem(tech_id=uuid.uuid4(), title="x")
    try:
        s.title = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("SimilarItem should be frozen")
