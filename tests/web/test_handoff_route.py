"""Route tests for POST /assets/{user_asset_id}/handoff (ARG-209).

Mirrors ``test_actions_route.py``'s style: the per-request session dependency
is overridden and ``transition_asset``/``_resolve_user_asset_tech_id`` are
monkeypatched so these run without a live database (release.yml CI has no
Postgres).
"""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.user_asset import AssetStatus
from argos.slack.services.asset_transition import TransitionOutcome
from argos.web.app import _get_session, build_web_app


class _FakeSession:
    """Minimal stand-in for an AsyncSession — the route commits explicitly."""

    def __init__(self) -> None:
        self.committed = False

    async def commit(self) -> None:
        self.committed = True


def _client(
    monkeypatch,
    *,
    verify_successor: bool = True,
    **patches,
) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session

    # The handoff route guards on ``_is_replace_successor`` before transitioning.
    # Default it to True (a valid Replace lineage) so the happy-path tests reach
    # the transitions; pass ``verify_successor=False`` to exercise rejection.
    async def _fake_verify(session, predecessor_id, successor_id):
        return verify_successor

    monkeypatch.setattr("argos.web.app._is_replace_successor", _fake_verify)

    # Whether the predecessor is a live Keep asset is now decided solely by the
    # locked transition_asset outcomes (no separate unlocked status read to
    # patch): each test's _fake_transition returns the outcome it wants to
    # exercise, so the ``_FakeSession`` never needs an ``execute``.

    # Detail-context handoff refreshes the 관련 신호 section out-of-band via
    # fetch_item_detail; default it to None (no OOB block) so the action-bar
    # tests don't need a full ItemDetailView. Overridable via ``patches``.
    async def _no_detail(session, tech_id):
        return None

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _no_detail)
    for path, fn in patches.items():
        monkeypatch.setattr(path, fn)
    return TestClient(app, raise_server_exceptions=False)


def test_handoff_archives_predecessor_and_keeps_successor(monkeypatch):
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()
    calls: list[tuple] = []

    async def _fake_resolve(session, ua_id):
        assert ua_id == user_asset_id
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        calls.append((tech_id, target_status))
        return TransitionOutcome.TRANSITIONED

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 200
    assert resp.text == ""
    assert calls == [
        (predecessor_tech_id, AssetStatus.ARCHIVED),
        (successor_tech_id, AssetStatus.KEEP),
    ]


def test_handoff_aborts_revival_when_predecessor_not_archived_from_live(monkeypatch):
    """codex P2 (race/revival): if the predecessor archive is a NOOP/CREATED (it
    was already Archived or gone — a concurrent Pass/Untrack, or a stale/crafted
    POST) but the successor Keep WOULD be a real new promotion, that revives a
    dismissed asset. The route must reject with 409 and NOT commit — so the
    pending successor Keep is discarded.

    (A completed-handoff replay, where BOTH transitions NOOP, is harmless and
    stays 200 — covered by test_handoff_replay_is_idempotent_still_returns_200.)
    """
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()
    sessions: list[_FakeSession] = []

    app = build_web_app()

    async def _fake_session():
        session = _FakeSession()
        sessions.append(session)
        yield session

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_verify(session, predecessor_id, successor_id):
        return True

    async def _fake_transition(session, tech_id, target_status):
        if tech_id == predecessor_tech_id:
            return TransitionOutcome.NOOP  # already Archived (raced) — no-op
        return TransitionOutcome.TRANSITIONED  # successor would be newly Kept

    monkeypatch.setattr("argos.web.app._resolve_user_asset_tech_id", _fake_resolve)
    monkeypatch.setattr("argos.web.app._is_replace_successor", _fake_verify)
    monkeypatch.setattr("argos.web.app.transition_asset", _fake_transition)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 409
    # Never committed → the pending successor Keep is rolled back on close.
    assert sessions and sessions[-1].committed is False


def test_handoff_rejects_recreated_predecessor_created_archive(monkeypatch):
    """codex P2: if the predecessor Keep row is DELETED after the live-status
    read (e.g. a stale feed ✓ Keep toggle-off), transition_asset(...ARCHIVED)
    returns CREATED — it would resurrect the predecessor as a phantom Archived
    row. Even when the successor is already Keep (kept == NOOP), that must be
    rejected (409, no commit): only a genuine Keep→Archived (TRANSITIONED), or a
    pure both-NOOP replay, may persist."""
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()
    sessions: list[_FakeSession] = []

    app = build_web_app()

    async def _fake_session():
        session = _FakeSession()
        sessions.append(session)
        yield session

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_verify(session, predecessor_id, successor_id):
        return True

    async def _no_detail(session, tech_id):
        return None

    async def _fake_transition(session, tech_id, target_status):
        if tech_id == predecessor_tech_id:
            return TransitionOutcome.CREATED  # row was deleted → resurrected
        return TransitionOutcome.NOOP  # successor already Keep

    monkeypatch.setattr("argos.web.app._resolve_user_asset_tech_id", _fake_resolve)
    monkeypatch.setattr("argos.web.app._is_replace_successor", _fake_verify)
    monkeypatch.setattr("argos.web.app.fetch_item_detail", _no_detail)
    monkeypatch.setattr("argos.web.app.transition_asset", _fake_transition)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 409
    assert sessions and sessions[-1].committed is False


def test_handoff_completed_replay_after_commit_returns_200(monkeypatch):
    """codex P2 (this fix): the REAL state after a handoff commits is predecessor
    Archived + successor Keep — so replaying the same POST (a detail double-submit
    or a stale back/forward) archives a NOOP and keeps a NOOP. An earlier unlocked
    ``_user_asset_status`` precheck saw the now-Archived predecessor and 409'd the
    replay before the both-NOOP outcome branch could recognize it as harmless,
    replacing a detail target with an error fragment. With that precheck gone the
    locked outcome gate sees both NOOP → benign replay → 200 (empty body)."""
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        # Post-commit replay: predecessor already Archived, successor already Keep.
        return TransitionOutcome.NOOP

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 200
    assert resp.text == ""


def test_handoff_unknown_asset_returns_404_fragment(monkeypatch):
    user_asset_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return None

    async def _fake_transition(session, tech_id, target_status):
        raise AssertionError("should not be called when the asset can't be resolved")

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 404
    assert "<!DOCTYPE html>" not in resp.text


def test_handoff_malformed_asset_uuid_returns_404(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/assets/not-a-uuid/handoff?successor_tech_id=" + str(uuid.uuid4()))
    assert resp.status_code == 404


def test_handoff_missing_successor_tech_id_returns_404(monkeypatch):
    user_asset_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        raise AssertionError("should not be called before successor_tech_id validation")

    client = _client(
        monkeypatch,
        **{"argos.web.app._resolve_user_asset_tech_id": _fake_resolve},
    )
    resp = client.post(f"/assets/{user_asset_id}/handoff")
    assert resp.status_code == 404


def test_handoff_malformed_successor_uuid_returns_404(monkeypatch):
    user_asset_id = uuid.uuid4()
    client = _client(monkeypatch)
    resp = client.post(f"/assets/{user_asset_id}/handoff?successor_tech_id=not-a-uuid")
    assert resp.status_code == 404


def test_handoff_replay_is_idempotent_still_returns_200(monkeypatch):
    """Both transitions NOOP on replay (transition_asset's own idempotency);
    the route still returns 200 with an empty body — no duplicate mutation,
    no error surfaced to a second click."""
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        return TransitionOutcome.NOOP

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    url = f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    first = client.post(url)
    second = client.post(url)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.text == second.text == ""


def test_handoff_already_kept_successor_still_archives_predecessor(monkeypatch):
    """AC4 "이미 tracking 중": the successor transition_asset call NOOPs (it
    was already Keep) but the predecessor archive still runs and the route
    still succeeds — 후속이 이미 Keep이어도 predecessor는 archive된다."""
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()
    calls: list[tuple] = []

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        calls.append((tech_id, target_status))
        if tech_id == successor_tech_id:
            return TransitionOutcome.NOOP  # already Keep
        return TransitionOutcome.TRANSITIONED  # Keep -> Archived

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 200
    assert (predecessor_tech_id, AssetStatus.ARCHIVED) in calls
    assert (successor_tech_id, AssetStatus.KEEP) in calls


def test_handoff_commits_session(monkeypatch):
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()
    sessions: list[_FakeSession] = []

    app = build_web_app()

    async def _fake_session():
        session = _FakeSession()
        sessions.append(session)
        yield session

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        return TransitionOutcome.TRANSITIONED

    async def _fake_verify(session, predecessor_id, successor_id):
        return True

    monkeypatch.setattr("argos.web.app._resolve_user_asset_tech_id", _fake_resolve)
    monkeypatch.setattr("argos.web.app.transition_asset", _fake_transition)
    monkeypatch.setattr("argos.web.app._is_replace_successor", _fake_verify)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 200
    assert sessions and sessions[-1].committed is True


def test_handoff_rejects_non_replace_successor(monkeypatch):
    """A ``successor_tech_id`` that is not an actual Replace successor of the
    predecessor (stale banner after the lineage changed, or a hand-crafted
    POST) is rejected with 409 and NOTHING transitions — the portfolio can't
    be handed off to an unrelated or self tech item."""
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        raise AssertionError("no transition may run when the successor is invalid")

    client = _client(
        monkeypatch,
        verify_successor=False,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 409
    assert "<!DOCTYPE html>" not in resp.text


def test_handoff_detail_context_rerenders_archived_action_bar(monkeypatch):
    """``context=detail`` re-renders the predecessor's action area, now
    Archived: the handoff banner is gone and the bar shows Keep (not a stale
    Untrack), swapped in via the ``#detail-actions-<id>`` wrapper so the detail
    page never displays controls for a state that no longer exists."""
    user_asset_id = uuid.uuid4()
    predecessor_tech_id = uuid.uuid4()
    successor_tech_id = uuid.uuid4()

    async def _fake_resolve(session, ua_id):
        return predecessor_tech_id

    async def _fake_transition(session, tech_id, target_status):
        return TransitionOutcome.TRANSITIONED

    async def _fake_lookup(session, tech_id):
        assert tech_id == predecessor_tech_id
        # Predecessor is Archived after the handoff; no ``successors`` key is
        # needed because the template short-circuits it when is_kept is False.
        return {
            "id": tech_id,
            "title": "Predecessor",
            "status": AssetStatus.ARCHIVED,
            "category": None,
            "image_url": None,
            "summary": None,
            "trust_score": None,
            "source_url": "https://x",
            "asset_id": user_asset_id,
        }

    client = _client(
        monkeypatch,
        **{
            "argos.web.app._resolve_user_asset_tech_id": _fake_resolve,
            "argos.web.app.transition_asset": _fake_transition,
            "argos.web.app._load_feed_card_context": _fake_lookup,
        },
    )
    resp = client.post(
        f"/assets/{user_asset_id}/handoff"
        f"?successor_tech_id={successor_tech_id}&context=detail"
    )
    assert resp.status_code == 200
    body = resp.text
    assert f'id="detail-actions-{predecessor_tech_id}"' in body
    assert f'id="item-actions-{predecessor_tech_id}"' in body
    assert "handoff-banner" not in body
    assert "Untrack" not in body
    assert "Keep" in body
