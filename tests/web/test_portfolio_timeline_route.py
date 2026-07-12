"""Route + fragment tests for GET /portfolio/{asset_id}/timeline (ARG-205).

Mirrors ``test_portfolio_route.py``'s style: the per-request session
dependency is overridden and ``fetch_timeline`` is monkeypatched so these
run without a live database (release.yml CI has no Postgres).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_succession import RelationType
from argos.web.app import _get_session, build_web_app
from argos.web.services.timeline import TimelineEvent


def _event(
    *,
    kind: str = "status",
    changed_at: datetime | None = None,
    title: str | None = None,
    link_tech_id: uuid.UUID | None = None,
    changed_from: str | None = None,
    changed_to: str | None = None,
    relation_type: RelationType | None = None,
    reasoning: str | None = None,
    label: str = "Tracking → Keep",
) -> TimelineEvent:
    return TimelineEvent(
        kind=kind,  # type: ignore[arg-type]
        changed_at=changed_at or datetime(2026, 6, 1, tzinfo=timezone.utc),
        title=title,
        link_tech_id=link_tech_id,
        changed_from=changed_from,
        changed_to=changed_to,
        relation_type=relation_type,
        reasoning=reasoning,
        label=label,
    )


def _client(
    monkeypatch,
    *,
    events: list[TimelineEvent] | None = None,
    tech_id: uuid.UUID | None = None,
    capture: list | None = None,
    raise_server_exceptions: bool = True,
) -> TestClient:
    """Build a TestClient whose timeline route returns ``events`` without DB access."""
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_resolve_tech_id(session, user_asset_id):
        return tech_id

    async def _fake_fetch_timeline(session, resolved_tech_id, *, limit=None):
        if capture is not None:
            capture.append({"tech_id": resolved_tech_id, "limit": limit})
        return events or []

    monkeypatch.setattr(
        "argos.web.app._resolve_user_asset_tech_id", _fake_resolve_tech_id
    )
    monkeypatch.setattr("argos.web.app.fetch_timeline", _fake_fetch_timeline)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


# ------------------------------------------------------------------ #
# Happy path
# ------------------------------------------------------------------ #

def test_get_timeline_returns_200_with_events(monkeypatch):
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    events = [
        _event(kind="status", changed_from="Tracking", changed_to="Keep", label="Tracking → Keep"),
        _event(
            kind="signal",
            title="Matched Item",
            link_tech_id=uuid.uuid4(),
            label="새 신호: Matched Item",
        ),
        _event(
            kind="succession",
            title="Successor Item",
            link_tech_id=uuid.uuid4(),
            relation_type=RelationType.ENHANCE,
            reasoning="better approach",
            label="Enhance: Successor Item",
        ),
    ]
    client = _client(monkeypatch, events=events, tech_id=tech_id)
    resp = client.get(f"/portfolio/{asset_id}/timeline")
    assert resp.status_code == 200
    body = resp.text
    assert "Tracking → Keep" in body
    assert "새 신호: Matched Item" in body
    assert "Enhance" in body
    assert "Successor Item" in body
    assert "better approach" in body


def test_get_timeline_fetches_with_limit_five(monkeypatch):
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    capture: list = []
    client = _client(monkeypatch, events=[], tech_id=tech_id, capture=capture)
    resp = client.get(f"/portfolio/{asset_id}/timeline")
    assert resp.status_code == 200
    assert capture and capture[-1]["tech_id"] == tech_id
    assert capture[-1]["limit"] == 5


def test_get_timeline_empty_events_renders_empty_state(monkeypatch):
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    client = _client(monkeypatch, events=[], tech_id=tech_id)
    body = client.get(f"/portfolio/{asset_id}/timeline").text
    assert "아직 추적 이벤트가 없습니다" in body


def test_get_timeline_fragment_is_partial_only(monkeypatch):
    """The fragment must not carry the full page shell (HTMX innerHTML swap)."""
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    client = _client(monkeypatch, events=[], tech_id=tech_id)
    body = client.get(f"/portfolio/{asset_id}/timeline").text
    assert "<!DOCTYPE html>" not in body
    assert 'class="tabbar"' not in body


def test_get_timeline_signal_links_to_matched_item(monkeypatch):
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    matched_id = uuid.uuid4()
    events = [
        _event(
            kind="signal",
            title="Linked Signal",
            link_tech_id=matched_id,
            label="새 신호: Linked Signal",
        )
    ]
    client = _client(monkeypatch, events=events, tech_id=tech_id)
    body = client.get(f"/portfolio/{asset_id}/timeline").text
    assert f'href="/item/{matched_id}"' in body


def test_get_timeline_legacy_succession_alert_has_no_link(monkeypatch):
    """title=None events (legacy succession_alerted) render as plain text,
    not a dangling <a href> with an empty target."""
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    events = [_event(kind="signal", title=None, link_tech_id=None, label="후속 기술 신호")]
    client = _client(monkeypatch, events=events, tech_id=tech_id)
    body = client.get(f"/portfolio/{asset_id}/timeline").text
    assert "후속 기술 신호" in body
    assert 'href="/item/None"' not in body


# ------------------------------------------------------------------ #
# Missing / invalid asset
# ------------------------------------------------------------------ #

def test_get_timeline_unknown_asset_returns_error_fragment(monkeypatch):
    """An asset_id with no matching user_asset row → controlled error
    fragment, not a 500."""
    asset_id = uuid.uuid4()
    client = _client(
        monkeypatch, events=[], tech_id=None, raise_server_exceptions=False
    )
    resp = client.get(f"/portfolio/{asset_id}/timeline")
    assert resp.status_code == 404
    assert "<!DOCTYPE html>" not in resp.text


def test_get_timeline_malformed_asset_id_returns_404_not_500(monkeypatch):
    client = _client(monkeypatch, events=[], tech_id=uuid.uuid4(), raise_server_exceptions=False)
    resp = client.get("/portfolio/not-a-uuid/timeline")
    assert resp.status_code == 404


# ------------------------------------------------------------------ #
# ARG-209 — Enhance/Fork "이 기술도 Keep" button + inferred 이어받음 line
# ------------------------------------------------------------------ #

def test_get_timeline_enhance_succession_shows_keep_button(monkeypatch):
    """Enhance/Fork successions (not Replace) get an inline "이 기술도 Keep"
    button reusing POST /items/{id}/keep — no banner (AC3)."""
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    succ_id = uuid.uuid4()
    events = [
        _event(
            kind="succession",
            title="Enhanced Successor",
            link_tech_id=succ_id,
            relation_type=RelationType.ENHANCE,
            label="Enhance: Enhanced Successor",
        )
    ]
    client = _client(monkeypatch, events=events, tech_id=tech_id)
    resp = client.get(f"/portfolio/{asset_id}/timeline")
    assert resp.status_code == 200
    body = resp.text
    assert "이 기술도 Keep" in body
    assert f'hx-post="/items/{succ_id}/keep"' in body
    assert "handoff-banner" not in body


def test_get_timeline_replace_succession_has_no_keep_button(monkeypatch):
    """A real forward Replace succession event (not yet handed off) renders
    plainly in the timeline — its CTA lives in the card-top banner, not a
    second button duplicated inside the timeline row."""
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    succ_id = uuid.uuid4()
    events = [
        _event(
            kind="succession",
            title="Replace Successor",
            link_tech_id=succ_id,
            relation_type=RelationType.REPLACE,
            label="Replace: Replace Successor",
        )
    ]
    client = _client(monkeypatch, events=events, tech_id=tech_id)
    resp = client.get(f"/portfolio/{asset_id}/timeline")
    assert resp.status_code == 200
    body = resp.text
    assert "Replace Successor" in body
    assert "이 기술도 Keep" not in body


def test_get_timeline_renders_inferred_handoff_line(monkeypatch):
    """The synthetic 이어받음 event (is_inferred=True) renders as plain
    informational text — no CTA button, since the handoff already happened."""
    tech_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    pred_id = uuid.uuid4()
    events = [
        _event(
            kind="succession",
            title="Old Predecessor",
            link_tech_id=pred_id,
            relation_type=RelationType.REPLACE,
            label="🔁 Old Predecessor에서 이어받음",
        )
    ]
    # _event() doesn't expose is_inferred; build directly for this one.
    from dataclasses import replace as _dc_replace

    events[0] = _dc_replace(events[0], is_inferred=True)

    client = _client(monkeypatch, events=events, tech_id=tech_id)
    resp = client.get(f"/portfolio/{asset_id}/timeline")
    assert resp.status_code == 200
    body = resp.text
    assert "이어받음" in body
    assert "Old Predecessor" in body
    assert "이어받기" not in body
    assert "이 기술도 Keep" not in body
    assert "timeline__event--inferred" in body
