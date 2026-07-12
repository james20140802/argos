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

    # Detail-context actions (?context=detail) refresh the 관련 신호 section
    # out-of-band via fetch_item_detail. Default it to None (no OOB block) so the
    # feed/action-bar tests don't need to build a full ItemDetailView; tests that
    # assert the signals refresh override this in ``patches``.
    async def _no_detail(session, tech_id):
        return None

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _no_detail)

    # A detail-context re-render that lands on Keep loads the handoff-banner
    # successors via _load_item_successors (centralized in _detail_action_response
    # so untrack's Keep fallback also gets the banner — codex P2). Default it to
    # empty (no banner) so tests that don't care about the banner never hit the
    # real DB loader; banner tests override this in ``patches``.
    async def _no_successors(session, tech_id):
        return []

    monkeypatch.setattr("argos.web.app._load_item_successors", _no_successors)
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
        # Mirror the real _load_feed_card_context 8-key shape (incl. summary,
        # trust_score)
        # so the partial re-renders the same data the production path would.
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "킵된 한 줄 요약",
                "trust_score": None, "source_url": "https://x"}

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


def test_keep_re_renders_card_with_trust_dial(monkeypatch):
    """A Keep/Pass re-render threads trust_score through the separate
    _load_feed_card_context dict path (not the FeedItem service path), so the
    compact trust dial must appear in the swapped-in fragment for an item that
    has a trust_score — the render-on-both-paths contract (ARG-189)."""
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Trusted Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": None,
                "trust_score": 0.6, "source_url": "https://x"}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 200
    # 0.6 → round(60) percent, rendered inside the compact eyebrow dial.
    assert "trust-dial" in resp.text
    assert "60" in resp.text


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
                "trust_score": None, "source_url": "https://x"}

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
                "trust_score": None, "source_url": "https://x"}

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
                "trust_score": None, "source_url": "https://x"}

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
                "trust_score": None, "source_url": "https://x"}

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


# --- ARG-184: ?context=detail re-renders the detail-page action bar ------ #


def test_keep_with_detail_context_returns_detail_actions_partial(monkeypatch):
    """``?context=detail`` (sent only by the item-detail page's action bar)
    must return the standalone ``_detail_actions.html`` fragment — never the
    feed's ``_feed_card.html`` — so a detail-page Keep click doesn't splice a
    full feed card into the single-item page."""
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "요약",
                "trust_score": None, "source_url": "https://x", "asset_id": uuid.uuid4()}

    async def _fake_successors(session, tech_id):
        return []

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
            "argos.web.app._load_item_successors": _fake_successors,
        },
    )
    resp = client.post(f"/items/{item_id}/keep?context=detail")
    assert resp.status_code == 200
    body = resp.text
    assert f'id="item-actions-{item_id}"' in body
    # The detail action bar never contains the feed card's markup.
    assert "headline" not in body
    assert "eyebrow" not in body
    assert "card--featured" not in body
    # Kept → Untrack replaces the Keep button.
    assert "Untrack" in body


def test_keep_with_detail_context_shows_handoff_banner_for_replace_successor(
    monkeypatch,
):
    """A Keep on the detail page for an item that already has a Replace
    successor must render the handoff banner in the swapped fragment — not omit
    it until a full reload. The re-render loads the item's successors (which
    _load_feed_card_context doesn't carry) so the banner appears immediately."""
    from argos.models.tech_succession import RelationType
    from argos.web.services.detail import GenealogyEntry

    item_id = uuid.uuid4()
    successor_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "요약",
                "trust_score": None, "source_url": "https://x", "asset_id": uuid.uuid4()}

    async def _fake_successors(session, tech_id):
        assert tech_id == item_id
        return [
            GenealogyEntry(
                tech_id=successor_id,
                title="Next-Gen",
                relation_type=RelationType.REPLACE,
                reasoning=None,
            )
        ]

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
            "argos.web.app._load_item_successors": _fake_successors,
        },
    )
    resp = client.post(f"/items/{item_id}/keep?context=detail")
    assert resp.status_code == 200
    body = resp.text
    assert "handoff-banner" in body
    assert "Next-Gen" in body
    assert f"successor_tech_id={successor_id}" in body


def test_keep_with_detail_context_refreshes_signals_oob(monkeypatch):
    """codex P2: a detail-page Keep must ALSO refresh the 관련 신호 section
    out-of-band — that section now shows the Keep-only unified timeline, which
    the action-bar swap alone would leave stale until a full reload. The
    response carries an hx-swap-oob wrapper (#detail-signals-<id>) rendering the
    timeline alongside the primary action-bar swap."""
    from datetime import datetime, timezone

    from argos.web.services.detail import ItemDetailView
    from argos.web.services.timeline import TimelineEvent

    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "요약",
                "trust_score": None, "source_url": "https://x", "asset_id": uuid.uuid4()}

    async def _fake_successors(session, tech_id):
        return []

    async def _fake_detail(session, tech_id):
        return ItemDetailView(
            id=item_id, title="Kept Thing", source_url="https://x",
            image_url=None, summary=None, category=None, trust_score=None,
            published_at=None, status=AssetStatus.KEEP, asset_id=uuid.uuid4(),
            timeline=[
                TimelineEvent(
                    kind="signal",
                    changed_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                    title="새 신호", link_tech_id=uuid.uuid4(),
                    changed_from=None, changed_to=None, relation_type=None,
                    reasoning=None, label="새 신호: X",
                )
            ],
        )

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
            "argos.web.app._load_item_successors": _fake_successors,
            "argos.web.app.fetch_item_detail": _fake_detail,
        },
    )
    resp = client.post(f"/items/{item_id}/keep?context=detail")
    assert resp.status_code == 200
    body = resp.text
    # primary swap: the action bar
    assert f'id="item-actions-{item_id}"' in body
    # out-of-band swap: the refreshed signals section with the Keep timeline
    assert f'id="detail-signals-{item_id}"' in body
    assert 'hx-swap-oob="true"' in body
    assert 'class="timeline"' in body


def test_pass_with_detail_context_returns_detail_actions_partial(monkeypatch):
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Passed", "status": AssetStatus.ARCHIVED,
                "category": None, "image_url": None, "summary": None,
                "trust_score": None, "source_url": "https://x", "asset_id": None}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/pass?context=detail")
    assert resp.status_code == 200
    body = resp.text
    assert f'id="item-actions-{item_id}"' in body
    assert "✓ Pass" in body
    assert "headline" not in body


def test_keep_without_context_still_returns_feed_card_partial(monkeypatch):
    """Plain (no ``context``) requests — i.e. every existing feed hx-post —
    keep returning the feed-card fragment. Guards ARG-184 against regressing
    the feed's Keep/Pass behavior."""
    item_id = uuid.uuid4()

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _fake_lookup(session, tech_id):
        return {"id": tech_id, "title": "Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": "요약",
                "trust_score": None, "source_url": "https://x", "asset_id": uuid.uuid4()}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _fake_toggle,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(f"/items/{item_id}/keep")
    assert resp.status_code == 200
    body = resp.text
    assert f'id="feed-card-{item_id}"' in body
    assert f'id="item-actions-{item_id}"' not in body


def test_untrack_with_detail_context_rerenders_action_bar(monkeypatch):
    """From the detail page, Untrack re-renders the action bar (now showing
    a plain Keep button again, since the asset moved to Archived) instead of
    returning an empty body — there is only one card on that page, so
    "removing" it like the portfolio does would leave no actions at all."""
    user_asset_id = uuid.uuid4()
    tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        assert ua_id == user_asset_id
        return tech_id

    async def _fake_transition(session, t_id, target_status):
        return TransitionOutcome.TRANSITIONED

    async def _fake_lookup(session, t_id):
        assert t_id == tech_id
        return {"id": t_id, "title": "Untracked", "status": AssetStatus.ARCHIVED,
                "category": None, "image_url": None, "summary": None,
                "trust_score": None, "source_url": "https://x", "asset_id": user_asset_id}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/untrack?context=detail&tech_id={tech_id}"
    )
    assert resp.status_code == 200
    body = resp.text
    assert body != ""
    assert f'id="item-actions-{tech_id}"' in body
    # No longer Keep, so the bar shows Keep (not Untrack) + an active Pass.
    assert "Untrack" not in body
    assert "✓ Pass" in body


def test_untrack_with_detail_context_falls_back_to_query_tech_id(monkeypatch):
    """A stale user_asset_id that no longer resolves (e.g. the row was already
    cleared) still re-renders the detail action bar using the ``tech_id``
    threaded through the URL, rather than erroring on a page that's still
    showing a live item."""
    user_asset_id = uuid.uuid4()
    tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return None

    async def _fake_lookup(session, t_id):
        assert t_id == tech_id
        return {"id": t_id, "title": "Still Here", "status": None,
                "category": None, "image_url": None, "summary": None,
                "trust_score": None, "source_url": "https://x", "asset_id": None}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/untrack?context=detail&tech_id={tech_id}"
    )
    assert resp.status_code == 200
    body = resp.text
    assert f'id="item-actions-{tech_id}"' in body
    assert "Untrack" not in body


def test_untrack_with_detail_context_falls_back_to_unrelated_tech_id(monkeypatch):
    """A stale ``user_asset_id`` (already-cleared, so it no longer resolves)
    combined with a ``tech_id`` query param pointing at a real but *unrelated*
    tech_item still renders that unrelated item's action bar — the fallback
    has no way to cross-check ``tech_id`` against the page the request
    originated from, since a cleared asset can no longer be resolved back to
    the original item. This pins that as the current (known) behavior.

    The one thing that must never happen is a state mutation running against
    the forged id: when ``resolved_tech_id`` is ``None``, ``transition_asset``
    is skipped entirely — the fallback only affects what gets *rendered*,
    never what gets *archived*. Assert that explicitly here rather than
    relying on it being implied by the resolve mock returning ``None``."""
    user_asset_id = uuid.uuid4()
    unrelated_tech_id = uuid.uuid4()
    transition_calls = []

    async def _fake_resolve(session, ua_id):
        return None

    async def _fake_transition(session, t_id, target_status):
        transition_calls.append(t_id)
        return TransitionOutcome.TRANSITIONED

    async def _fake_lookup(session, t_id):
        assert t_id == unrelated_tech_id
        return {"id": t_id, "title": "Unrelated Item", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": None,
                "trust_score": None, "source_url": "https://unrelated", "asset_id": uuid.uuid4()}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/untrack?context=detail&tech_id={unrelated_tech_id}"
    )

    # No mutation ran against the forged/unrelated id.
    assert transition_calls == []
    assert resp.status_code == 200
    body = resp.text
    # Known behavior: the fragment renders for the unrelated tech_id, not the
    # item the page was originally showing — this is the DOM id mismatch the
    # fallback can produce when tech_id doesn't match the original page.
    assert f'id="item-actions-{unrelated_tech_id}"' in body
    assert "Untrack" in body  # AssetStatus.KEEP renders the Untrack button


def test_untrack_detail_fallback_on_keep_item_renders_handoff_banner(monkeypatch):
    """codex P2: when untrack's ``user_asset_id`` no longer resolves, it falls
    back to rendering the live asset for the ``tech_id`` query param. If that
    item is currently Keep AND has a Replace successor, the swapped action bar
    must still show the handoff banner. The banner filters on ``item.successors``,
    which ``_load_feed_card_context`` omits — earlier only the keep/pass path
    pre-loaded them, so this fallback silently dropped the banner (Jinja treats
    the missing key as falsy, no error). _detail_action_response now loads them
    for every Keep detail re-render, so the banner appears here too."""
    from argos.models.tech_succession import RelationType
    from argos.web.services.detail import GenealogyEntry

    user_asset_id = uuid.uuid4()
    live_tech_id = uuid.uuid4()
    successor_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return None  # stale user_asset_id — triggers the tech_id fallback

    async def _fake_lookup(session, t_id):
        assert t_id == live_tech_id
        return {"id": t_id, "title": "Re-Kept Thing", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": None,
                "trust_score": None, "source_url": "https://x", "asset_id": uuid.uuid4()}

    async def _fake_successors(session, t_id):
        assert t_id == live_tech_id
        return [
            GenealogyEntry(
                tech_id=successor_id,
                title="Next-Gen",
                relation_type=RelationType.REPLACE,
                reasoning=None,
            )
        ]

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app._load_feed_card_context": _fake_lookup,
            "argos.web.app._load_item_successors": _fake_successors,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/untrack?context=detail&tech_id={live_tech_id}"
    )
    assert resp.status_code == 200
    body = resp.text
    assert "handoff-banner" in body
    assert f"successor_tech_id={successor_id}" in body


def test_untrack_with_detail_context_and_no_resolvable_tech_id_404s(monkeypatch):
    user_asset_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return None

    client = _client(
        monkeypatch,
        **{"argos.web.app._resolve_user_asset_tech_id": _fake_resolve},
    )
    resp = client.post(f"/assets/{user_asset_id}/untrack?context=detail")
    assert resp.status_code == 404


def test_untrack_without_context_still_returns_empty_body(monkeypatch):
    """Plain (no ``context``) untrack — i.e. the portfolio's hx-post — keeps
    returning an empty 200 so its outerHTML swap removes the card, unchanged
    by ARG-184."""
    user_asset_id = uuid.uuid4()
    tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return tech_id

    async def _fake_transition(session, t_id, target_status):
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
