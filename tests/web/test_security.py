"""ARG-195: login + CSRF gate on the real web app (no DB).

Every test here runs with the *enforced* policy (``real_security`` marker) and
a tmp token file — see ``tests/web/conftest.py``.
"""
from __future__ import annotations

import uuid

import pytest
from starlette.testclient import TestClient

from argos.web import auth
from argos.web.app import _get_session, build_web_app

pytestmark = pytest.mark.real_security

BASE = "http://testserver"
SAME_ORIGIN = {"Sec-Fetch-Site": "same-origin"}


class _FakeSession:
    async def commit(self) -> None:
        return None

    async def execute(self, *a, **k):  # pragma: no cover - not reached
        raise AssertionError("DB must not be touched")


def _app(monkeypatch, calls: list | None = None):
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session
    calls = calls if calls is not None else []

    async def _reached_handler(session, tech_id):
        calls.append(tech_id)
        raise RuntimeError("stop after handler reached")  # reached = authorised

    async def _fake_toggle(session, tech_id, target_status, *, currently_active=False):
        calls.append(("toggle", tech_id))
        raise RuntimeError("stop after service call")

    # The Keep handler loads the card first, then toggles: either one being
    # reached means the request got past the gate.
    monkeypatch.setattr("argos.web.app._load_feed_card_context", _reached_handler)
    monkeypatch.setattr("argos.web.app.toggle_asset", _fake_toggle)
    return app


def _client(monkeypatch, calls=None, *, authed_token: str | None = None) -> TestClient:
    client = TestClient(_app(monkeypatch, calls), raise_server_exceptions=False)
    if authed_token is not None:
        client.cookies.set(auth.SESSION_COOKIE, auth.session_value(authed_token))
    return client


# --- unauthenticated -------------------------------------------------------


def test_html_get_without_cookie_renders_login_inline_401(monkeypatch):
    r = _client(monkeypatch).get(
        "/feed", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert r.status_code == 401
    assert "location" not in r.headers
    assert 'action="/login"' in r.text
    assert 'name="next" value="/feed"' in r.text
    assert r.headers["cache-control"] == "no-store"


def test_htmx_without_cookie_gets_hx_redirect(monkeypatch):
    item = uuid.uuid4()
    r = _client(monkeypatch).post(
        f"/items/{item}/keep",
        headers={
            **SAME_ORIGIN,
            "HX-Request": "true",
            "HX-Current-URL": f"{BASE}/portfolio?x=1",
        },
    )
    assert r.status_code == 401
    assert r.headers["HX-Redirect"] == "/login?next=/portfolio%3Fx%3D1"


def test_post_keep_without_cookie_is_401_and_service_not_called(monkeypatch):
    calls: list = []
    r = _client(monkeypatch, calls).post(f"/items/{uuid.uuid4()}/keep", headers=SAME_ORIGIN)
    assert r.status_code == 401
    assert calls == []


def test_json_get_without_cookie_is_401_json(monkeypatch):
    r = _client(monkeypatch).get("/feed/poll")
    assert r.status_code == 401
    assert r.json() == {"detail": "authentication required"}


def test_forged_tailscale_identity_header_is_ignored(monkeypatch):
    r = _client(monkeypatch).get(
        "/feed",
        headers={
            "Accept": "text/html",
            "Tailscale-User-Login": "owner@example.com",
            "Tailscale-User-Name": "Owner",
        },
    )
    assert r.status_code == 401


def test_cookie_from_other_token_is_rejected(monkeypatch):
    calls: list = []
    c = _client(monkeypatch, calls, authed_token="not-the-real-token")
    r = c.post(f"/items/{uuid.uuid4()}/keep", headers=SAME_ORIGIN)
    assert r.status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    "path",
    [
        "/healthz",
        "/manifest.webmanifest",
        "/sw.js",
        "/login",
        "/static/css/argos.css",
        "/static/img/icons/icon-192.png",
    ],
)
def test_exempt_paths_reachable_without_cookie(monkeypatch, path):
    r = _client(monkeypatch).get(path, follow_redirects=False)
    assert r.status_code == 200, path


# --- CSRF -------------------------------------------------------------------


def test_authed_cross_origin_post_is_403(monkeypatch, _web_security_policy):
    calls: list = []
    c = _client(monkeypatch, calls, authed_token=_web_security_policy["token"])
    r = c.post(f"/items/{uuid.uuid4()}/keep", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert calls == []


def test_authed_sec_fetch_site_cross_site_is_403(monkeypatch, _web_security_policy):
    calls: list = []
    c = _client(monkeypatch, calls, authed_token=_web_security_policy["token"])
    r = c.post(
        f"/items/{uuid.uuid4()}/keep",
        headers={"Sec-Fetch-Site": "cross-site", "Origin": BASE},
    )
    assert r.status_code == 403
    assert calls == []


def test_authed_post_without_origin_or_fetch_metadata_is_403(monkeypatch, _web_security_policy):
    c = _client(monkeypatch, authed_token=_web_security_policy["token"])
    r = c.post(f"/items/{uuid.uuid4()}/keep")
    assert r.status_code == 403


def test_authed_same_origin_post_reaches_service(monkeypatch, _web_security_policy):
    calls: list = []
    c = _client(monkeypatch, calls, authed_token=_web_security_policy["token"])
    item = uuid.uuid4()
    c.post(f"/items/{item}/keep", headers={"Origin": BASE})
    assert calls == [item]


def test_authed_same_origin_via_fetch_metadata_reaches_service(monkeypatch, _web_security_policy):
    calls: list = []
    c = _client(monkeypatch, calls, authed_token=_web_security_policy["token"])
    item = uuid.uuid4()
    c.post(f"/items/{item}/keep", headers=SAME_ORIGIN)
    assert calls == [item]


def test_x_forwarded_host_match_is_accepted(monkeypatch, _web_security_policy):
    """tailscale serve: Origin is the ts.net name, Host may be 127.0.0.1:8765."""
    calls: list = []
    c = _client(monkeypatch, calls, authed_token=_web_security_policy["token"])
    item = uuid.uuid4()
    c.post(
        f"/items/{item}/keep",
        headers={
            "Host": "127.0.0.1:8765",
            "X-Forwarded-Host": "mac.tail1234.ts.net",
            "Origin": "https://mac.tail1234.ts.net",
        },
    )
    assert calls == [item]


def test_events_batch_beacon_passes(monkeypatch, _web_security_policy):
    """sendBeacon: text/plain body, Origin + Sec-Fetch-Site same-origin, cookie."""
    c = _client(monkeypatch, authed_token=_web_security_policy["token"])
    r = c.post(
        "/events/batch",
        content=b'{"events": []}',
        headers={"Content-Type": "text/plain;charset=UTF-8", "Origin": BASE, **SAME_ORIGIN},
    )
    assert r.status_code == 200
    assert r.json()["inserted"] == 0


def test_settings_post_cross_site_is_403(monkeypatch, _web_security_policy, tmp_path):
    c = _client(monkeypatch, authed_token=_web_security_policy["token"])
    r = c.post("/settings", data={"slack.channel_id": "C1"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


# --- login flow -------------------------------------------------------------


def test_login_wrong_token_is_401(monkeypatch):
    r = _client(monkeypatch).post(
        "/login", data={"token": "nope", "next": "/portfolio"}, headers=SAME_ORIGIN
    )
    assert r.status_code == 401
    assert auth.SESSION_COOKIE not in r.headers.get("set-cookie", "")
    assert "토큰이 맞지 않습니다" in r.text


def test_login_right_token_sets_cookie_and_redirects(monkeypatch, _web_security_policy):
    token = _web_security_policy["token"]
    r = _client(monkeypatch).post(
        "/login",
        data={"token": token, "next": "/portfolio"},
        headers=SAME_ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/portfolio"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{auth.SESSION_COOKIE}={auth.session_value(token)};")
    lowered = cookie.lower()
    assert "httponly" in lowered
    assert "samesite=lax" in lowered
    assert "max-age=34560000" in lowered
    assert "path=/" in lowered
    assert "secure" not in lowered  # plain http test server
    assert token not in cookie


def test_login_cookie_secure_behind_https_proxy(monkeypatch, _web_security_policy):
    r = _client(monkeypatch).post(
        "/login",
        data={"token": _web_security_policy["token"]},
        headers={**SAME_ORIGIN, "X-Forwarded-Proto": "https"},
        follow_redirects=False,
    )
    assert "secure" in r.headers["set-cookie"].lower()


def test_login_post_cross_site_is_403(monkeypatch, _web_security_policy):
    r = _client(monkeypatch).post(
        "/login",
        data={"token": _web_security_policy["token"]},
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert r.status_code == 403


def test_login_link_with_token_sets_cookie_and_strips_token(monkeypatch, _web_security_policy):
    token = _web_security_policy["token"]
    r = _client(monkeypatch).get(
        f"/login?token={token}&next=/portfolio", follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/portfolio"
    assert token not in r.headers["location"]
    assert r.headers["referrer-policy"] == "no-referrer"
    assert auth.SESSION_COOKIE in r.headers["set-cookie"]


def test_login_link_with_wrong_token_is_401(monkeypatch):
    r = _client(monkeypatch).get("/login?token=wrong", follow_redirects=False)
    assert r.status_code == 401
    assert "set-cookie" not in r.headers


@pytest.mark.parametrize(
    "bad", ["//evil.example/x", "https://evil.example", "/\\evil.example", "evil", "/login"]
)
def test_login_rejects_offsite_next(monkeypatch, _web_security_policy, bad):
    r = _client(monkeypatch).post(
        "/login",
        data={"token": _web_security_policy["token"], "next": bad},
        headers=SAME_ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/feed"


def test_after_login_feed_is_reachable(monkeypatch, _web_security_policy):
    c = _client(monkeypatch)
    c.post(
        "/login",
        data={"token": _web_security_policy["token"], "next": "/healthz"},
        headers=SAME_ORIGIN,
    )
    r = c.get("/feed/poll", follow_redirects=False)
    assert r.status_code != 401


def test_rotation_invalidates_existing_cookie(monkeypatch, _web_security_policy):
    c = _client(monkeypatch, authed_token=_web_security_policy["token"])
    assert c.get("/feed/poll").status_code != 401
    auth.rotate_token(_web_security_policy["token_path"])
    assert c.get("/feed/poll").status_code == 401


def test_logout_clears_cookie(monkeypatch, _web_security_policy):
    c = _client(monkeypatch, authed_token=_web_security_policy["token"])
    r = c.post("/logout", headers=SAME_ORIGIN, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
    assert f'{auth.SESSION_COOKIE}=""' in r.headers["set-cookie"]


def test_missing_token_file_fails_closed(monkeypatch, _web_security_policy):
    _web_security_policy["token_path"].unlink()
    c = _client(monkeypatch, authed_token=_web_security_policy["token"])
    assert c.get("/feed/poll").status_code == 401
    r = c.post("/login", data={"token": _web_security_policy["token"]}, headers=SAME_ORIGIN)
    assert r.status_code == 401
