"""Route tests for Keep/Pass/Untrack HTMX actions (ARG-139)."""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.user_asset import AssetStatus
from argos.slack.services.asset_transition import TransitionOutcome
from argos.web.app import _get_session, build_web_app


def _client(monkeypatch, **patches) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session
    for path, fn in patches.items():
        monkeypatch.setattr(path, fn)
    return TestClient(app, raise_server_exceptions=False)


def test_keep_returns_updated_feed_card_partial(monkeypatch):
    item_id = uuid.uuid4()
    captured: dict = {}

    async def _fake_transition(session, tech_id, target_status):
        captured["tech_id"] = tech_id
        captured["target_status"] = target_status
        return TransitionOutcome.CREATED

    async def _fake_lookup(session, tech_id):
        # Returns a tiny shape sufficient for the partial.
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 200
    assert "Kept Thing" in resp.text
    # Partial, not full page.
    assert "<!DOCTYPE html>" not in resp.text
    assert captured["target_status"] == AssetStatus.KEEP
    assert captured["tech_id"] == item_id


def test_pass_returns_updated_feed_card_partial(monkeypatch):
    item_id = uuid.uuid4()
    captured: dict = {}

    async def _fake_transition(session, tech_id, target_status):
        captured["target_status"] = target_status
        return TransitionOutcome.TRANSITIONED

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Passed", "status": AssetStatus.ARCHIVED,
                "category": None, "image_url": None, "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/pass")
    assert resp.status_code == 200
    assert "Passed" in resp.text
    assert captured["target_status"] == AssetStatus.ARCHIVED


def test_keep_unknown_item_returns_404_fragment(monkeypatch):
    item_id = uuid.uuid4()

    async def _fake_transition(session, tech_id, target_status):
        # transition_asset would raise IntegrityError on bad FK, simulate via
        # lookup miss before calling transition.
        raise AssertionError("should not be called")

    async def _fake_lookup_missing(session, tech_id):
        return None

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup_missing,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 404
    # 1-line fragment, not a full page.
    assert "<!DOCTYPE html>" not in resp.text


def test_keep_malformed_uuid_returns_404(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/items/not-a-uuid/keep")
    assert resp.status_code == 404


def test_keep_noop_returns_409_fragment(monkeypatch):
    item_id = uuid.uuid4()

    async def _fake_transition(session, tech_id, target_status):
        return TransitionOutcome.NOOP

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Already Kept", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 409
    assert "<!DOCTYPE html>" not in resp.text


def test_untrack_returns_updated_portfolio_row_partial(monkeypatch):
    user_asset_id = uuid.uuid4()
    tech_id = uuid.uuid4()
    captured: dict = {}

    async def _fake_resolve(session, ua_id):
        assert ua_id == user_asset_id
        return tech_id

    async def _fake_transition(session, t_id, target_status):
        captured["tech_id"] = t_id
        captured["target_status"] = target_status
        return TransitionOutcome.TRANSITIONED

    async def _fake_lookup(session, ua_id):
        return {"id": ua_id, "title": "Tracked", "status": AssetStatus.ARCHIVED,
                "category": None, "image_url": None}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_portfolio_row_context": _fake_lookup,
        },
    )
    resp = client.post(f"/assets/{user_asset_id}/untrack")
    assert resp.status_code == 200
    assert "Tracked" in resp.text
    assert "<!DOCTYPE html>" not in resp.text
    assert captured["target_status"] == AssetStatus.ARCHIVED
    assert captured["tech_id"] == tech_id


def test_untrack_unknown_asset_returns_404_fragment(monkeypatch):
    user_asset_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return None

    client = _client(
        monkeypatch,
        **{"argos.web.app._resolve_user_asset_tech_id": _fake_resolve},
    )
    resp = client.post(f"/assets/{user_asset_id}/untrack")
    assert resp.status_code == 404


def test_untrack_malformed_uuid_returns_404(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/assets/not-a-uuid/untrack")
    assert resp.status_code == 404


def test_untrack_already_archived_returns_409(monkeypatch):
    user_asset_id = uuid.uuid4()
    tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return tech_id

    async def _fake_transition(session, t_id, target_status):
        return TransitionOutcome.NOOP

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(f"/assets/{user_asset_id}/untrack")
    assert resp.status_code == 409
