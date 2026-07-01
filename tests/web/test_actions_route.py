"""Route tests for Keep/Pass/Untrack HTMX actions (ARG-139)."""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.user_asset import AssetStatus
from argos.slack.services.asset_transition import ToggleOutcome, TransitionOutcome
from argos.web.app import _get_session, build_web_app


class _FakeSession:
    """Minimal stand-in for an AsyncSession.

    The action routes commit the transition explicitly (the real
    ``get_session`` does not auto-commit), so the fake must accept an
    awaitable ``commit()``. All DB access itself is monkeypatched out.
    """

    async def commit(self) -> None:
        return None


def _client(monkeypatch, **patches) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session
    for path, fn in patches.items():
        monkeypatch.setattr(path, fn)
    return TestClient(app, raise_server_exceptions=False)


def test_keep_returns_updated_feed_card_partial(monkeypatch):
    item_id = uuid.uuid4()
    captured: dict = {}

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        captured["tech_id"] = tech_id
        captured["target_status"] = target_status
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        # Mirror the real _load_feed_card_context 7-key shape (incl. summary)
        # so the partial re-renders the same data the production path would.
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "킵된 한 줄 요약",
                "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 200
    assert "Kept Thing" in resp.text
    # The Keep button reflects the active state (ARG-... toggle contract).
    assert "✓ Keep" in resp.text
    # The summary line survives the action re-render (ARG-174/175 contract).
    assert "킵된 한 줄 요약" in resp.text
    # Partial, not full page.
    assert "<!DOCTYPE html>" not in resp.text
    assert captured["target_status"] == AssetStatus.KEEP
    assert captured["tech_id"] == item_id


def test_keep_on_featured_card_re_renders_as_hero(monkeypatch):
    """A Keep/Pass on the featured hero passes ?featured=1, so the swapped-in
    card must re-render as the hero (card--featured) rather than collapsing to
    a standard grid cell (ARG-175)."""
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Hero Kept", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "히어로 요약",
                "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    featured = client.post(f"/items/{item_id}/keep?featured=1")
    assert featured.status_code == 200
    assert "card--featured" in featured.text
    # Without the flag the same card re-renders as a standard (non-hero) card.
    plain = client.post(f"/items/{item_id}/keep")
    assert "card--featured" not in plain.text


def test_pass_returns_updated_feed_card_partial(monkeypatch):
    item_id = uuid.uuid4()
    captured: dict = {}

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        captured["target_status"] = target_status
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Passed", "status": AssetStatus.ARCHIVED,
                "category": None, "image_url": None, "summary": None,
                "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/pass")
    assert resp.status_code == 200
    assert "Passed" in resp.text
    assert "✓ Pass" in resp.text
    assert captured["target_status"] == AssetStatus.ARCHIVED


def test_keep_unknown_item_returns_404_fragment(monkeypatch):
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        # The lookup miss short-circuits to 404 before any toggle runs.
        raise AssertionError("should not be called")

    async def _fake_lookup_missing(session, tech_id):
        return None

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
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


def test_keep_again_toggles_off_returns_untriaged_card(monkeypatch):
    """Pressing Keep on an already-Kept item toggles it OFF: the service reports
    REMOVED and the reloaded card comes back untriaged (200, no active state) —
    not the old 409 "already in that state"."""
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.REMOVED

    async def _fake_lookup(session, tech_id):
        # After toggle-off there is no user_asset, so status is None.
        return {"id": tech_id, "title": "Untriaged Again", "status": None,
                "category": None, "image_url": None, "summary": None,
                "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 200
    assert "Untriaged Again" in resp.text
    # The Keep button is back to its inactive label — no check, no active class.
    assert "✓ Keep" not in resp.text
    assert "is-active" not in resp.text
    assert "<!DOCTYPE html>" not in resp.text


def test_active_query_param_threads_currently_active(monkeypatch):
    """``?active=1`` (the button was rendered pressed) reaches ``toggle_asset``
    as ``currently_active=True`` so the click is treated as a *clear*, not a
    blind re-toggle — the guard against stale service-worker-cached cards."""
    item_id = uuid.uuid4()
    captured: dict = {}

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        captured["currently_active"] = currently_active
        return ToggleOutcome.REMOVED

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Cleared", "status": None,
                "category": None, "image_url": None, "summary": None,
                "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep?active=1")
    assert resp.status_code == 200
    assert captured["currently_active"] is True

    # Absent param → inactive button → set intent.
    client.post(f"/items/{item_id}/keep")
    assert captured["currently_active"] is False


def test_untrack_returns_empty_body_to_remove_card(monkeypatch):
    # Untracking archives the asset, so it leaves the Keep-only portfolio.
    # The route returns an empty 200 body and the HTMX outerHTML swap deletes
    # the card client-side.
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

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(f"/assets/{user_asset_id}/untrack")
    assert resp.status_code == 200
    assert resp.text == ""
    assert captured["target_status"] == AssetStatus.ARCHIVED
    assert captured["tech_id"] == tech_id


def test_untrack_unknown_asset_removes_stale_card(monkeypatch):
    """A stale portfolio card whose asset was already cleared (missing row)
    must be *removed* (empty 200), not error — the untracked state already
    holds, so a 404 fragment would leave a dead card showing an error."""
    user_asset_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return None

    client = _client(
        monkeypatch,
        **{"argos.web.app._resolve_user_asset_tech_id": _fake_resolve},
    )
    resp = client.post(f"/assets/{user_asset_id}/untrack")
    assert resp.status_code == 200
    assert resp.text == ""


def test_untrack_malformed_uuid_returns_404(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/assets/not-a-uuid/untrack")
    assert resp.status_code == 404


def test_untrack_already_archived_removes_stale_card(monkeypatch):
    """A stale card for an asset already Archived (NOOP) is also removed: it is
    no longer a live Keep, so the Keep-only portfolio card is stale. Empty 200
    removes it idempotently rather than surfacing a 409 on a dead card."""
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
    assert resp.status_code == 200
    assert resp.text == ""
