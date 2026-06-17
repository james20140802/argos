"""Route tests for the 🔭 관련 신호 → 새 신호 (signal alerts) subsection.

The detail page resolves the Slack signal-alert rows (the same rows the
portfolio counts to mark a card active) into linked entries, so an active
card's detail page explains why it signalled rather than rendering raw
sentinel/UUID noise.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import ItemDetailView, SignalAlert, SimilarItem


def _view_with(signal_alerts=None, similar=None, related_history=None) -> ItemDetailView:
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
        signal_alerts=signal_alerts or [],
        related_history=related_history or [],
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


def test_new_signals_section_omitted_when_empty(monkeypatch):
    view = _view_with(signal_alerts=[])
    client = _client(monkeypatch, view)
    body = client.get(f"/item/{view.id}").text
    assert "새 신호" not in body
    assert "signals-new" not in body


def test_signal_match_renders_resolved_link(monkeypatch):
    matched_id = uuid.uuid4()
    view = _view_with(
        signal_alerts=[
            SignalAlert(
                kind="signal",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
                matched_tech_id=matched_id,
                matched_title="Mistral-Embed v2",
            )
        ]
    )
    client = _client(monkeypatch, view)
    body = client.get(f"/item/{view.id}").text
    assert "관련 신호" in body
    assert "새 신호" in body
    assert "Mistral-Embed v2" in body
    assert f"/item/{matched_id}" in body


def test_succession_alert_renders_generic_label(monkeypatch):
    view = _view_with(
        signal_alerts=[
            SignalAlert(
                kind="succession",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
            )
        ]
    )
    client = _client(monkeypatch, view)
    body = client.get(f"/item/{view.id}").text
    assert "후속 기술 신호" in body


def test_unresolved_signal_match_does_not_render_uuid(monkeypatch):
    """A signal whose matched item was deleted renders a generic label, never
    a raw UUID (the bug this whole subsection replaced)."""
    view = _view_with(
        signal_alerts=[
            SignalAlert(
                kind="signal",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
                matched_tech_id=None,
                matched_title=None,
            )
        ]
    )
    client = _client(monkeypatch, view)
    body = client.get(f"/item/{view.id}").text
    assert "새 신호" in body
    assert "signal_matched" not in body
    assert "None" not in body


def test_outer_section_renders_with_only_signal_alerts(monkeypatch):
    """관련 신호 wrapper appears when only the 새 신호 subsection has content."""
    view = _view_with(
        signal_alerts=[
            SignalAlert(
                kind="signal",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
                matched_tech_id=uuid.uuid4(),
                matched_title="Solo Signal",
            )
        ],
        similar=[],
        related_history=[],
    )
    client = _client(monkeypatch, view)
    body = client.get(f"/item/{view.id}").text
    assert "관련 신호" in body
    assert "Solo Signal" in body


def test_subsection_order_similar_then_new_then_history(monkeypatch):
    """Approved layout: 유사 신호 → 새 신호 → 최근 변화."""
    from argos.web.services.detail import HistoryEntry

    view = _view_with(
        similar=[SimilarItem(tech_id=uuid.uuid4(), title="SimItem")],
        signal_alerts=[
            SignalAlert(
                kind="signal",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
                matched_tech_id=uuid.uuid4(),
                matched_title="NewSig",
            )
        ],
        related_history=[
            HistoryEntry(
                changed_from="Tracking",
                changed_to="Keep",
                changed_at=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                tech_id=uuid.uuid4(),
                tech_title="HistItem",
            )
        ],
    )
    client = _client(monkeypatch, view)
    body = client.get(f"/item/{view.id}").text
    assert body.index("유사 신호") < body.index("새 신호") < body.index("최근 변화")
