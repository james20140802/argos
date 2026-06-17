"""Route tests for 🔭 track_history timeline on /item/{id} (ARG-161)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import (
    HistoryEntry,
    ItemDetailView,
    SimilarItem,
)


def _view_with(
    *,
    related_history: list[HistoryEntry] | None = None,
    similar: list[SimilarItem] | None = None,
) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title="Anchor",
        source_url="https://example.com/anchor",
        image_url=None,
        summary=None,
        category=None,
        trust_score=None,
        published_at=None,
        related_history=related_history or [],
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


def test_related_signals_section_renders_when_only_history_present(monkeypatch):
    tech_id = uuid.uuid4()
    view = _view_with(
        related_history=[
            HistoryEntry(
                changed_from="Tracking",
                changed_to="Keep",
                changed_at=datetime(2026, 6, 10, 9, 30, tzinfo=timezone.utc),
                tech_id=tech_id,
                tech_title="Claude Opus 4.7",
            )
        ]
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    # Outer 관련 신호 heading must appear even though similarity is empty.
    assert "관련 신호" in body
    assert "최근 변화" in body
    assert "Tracking" in body
    assert "Keep" in body
    assert "Claude Opus 4.7" in body
    assert "2026-06-10" in body
    assert f"/item/{tech_id}" in body


def test_history_subsection_omitted_when_history_empty(monkeypatch):
    view = _view_with(related_history=[])
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    # No subsection markup at all when empty.
    assert "최근 변화" not in resp.text
    assert "signals-history" not in resp.text


def test_history_and_similarity_both_render(monkeypatch):
    tech_id = uuid.uuid4()
    sim_id = uuid.uuid4()
    view = _view_with(
        related_history=[
            HistoryEntry(
                changed_from="Tracking",
                changed_to="Keep",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
                tech_id=tech_id,
                tech_title="LangGraph",
            )
        ],
        similar=[SimilarItem(tech_id=sim_id, title="LangChain-next")],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "관련 신호" in body
    assert "최근 변화" in body
    assert "유사 신호" in body
    assert "LangGraph" in body
    assert "LangChain-next" in body


def test_history_entries_render_in_provided_order(monkeypatch):
    """Service is responsible for ordering desc by changed_at; the template
    must render in that order (no re-sorting)."""
    older = HistoryEntry(
        changed_from="Tracking",
        changed_to="Keep",
        changed_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        tech_id=uuid.uuid4(),
        tech_title="Older",
    )
    newer = HistoryEntry(
        changed_from="Keep",
        changed_to="Archived",
        changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
        tech_id=uuid.uuid4(),
        tech_title="Newer",
    )
    # Service hands the template a desc-ordered list (newest first).
    view = _view_with(related_history=[newer, older])
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")
    body = resp.text
    assert resp.status_code == 200
    assert body.index("Newer") < body.index("Older")
