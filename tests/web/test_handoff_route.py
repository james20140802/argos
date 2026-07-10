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


def _client(monkeypatch, **patches) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session
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

    monkeypatch.setattr("argos.web.app._resolve_user_asset_tech_id", _fake_resolve)
    monkeypatch.setattr("argos.web.app.transition_asset", _fake_transition)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        f"/assets/{user_asset_id}/handoff?successor_tech_id={successor_tech_id}"
    )
    assert resp.status_code == 200
    assert sessions and sessions[-1].committed is True
