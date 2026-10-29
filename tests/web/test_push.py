"""ARG-321: web push plumbing, routes and the settings toggle — no DB needed.

Routes get a stub session (dependency override) and the service functions
they call are monkeypatched, so these run in release CI without Postgres.
The VAPID key always lives under ``tmp_path`` (``ARGOS_WEB_PUSH_KEY_FILE``).
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from argos.private_files import file_mode
from argos.web import push
from argos.web.app import _get_session, build_web_app
from argos.web.services import push_briefing

ENDPOINT = "https://web.push.apple.com/QGx1c3Q"
KEYS = {"p256dh": "BPpub", "auth": "secret"}


@pytest.fixture(autouse=True)
def _key_file(tmp_path, monkeypatch):
    path = tmp_path / "argos" / "web_push_vapid.pem"
    monkeypatch.setenv(push.KEY_FILE_ENV, str(path))
    return path


@pytest.fixture
def client():
    app = build_web_app()

    async def _stub_session():
        yield SimpleNamespace()

    app.dependency_overrides[_get_session] = _stub_session
    return TestClient(app)


# ---- keys & labels ----------------------------------------------------------


def test_vapid_key_is_created_owner_only_and_reused(_key_file):
    first = push.load_or_create_vapid()
    assert _key_file.exists()
    assert file_mode(_key_file) == 0o600

    again = push.load_or_create_vapid()
    assert push.public_key_b64(first) == push.public_key_b64(again)


def test_public_key_is_uncompressed_p256_base64url():
    key = push.public_key_b64(push.load_or_create_vapid())
    raw = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4))
    assert len(raw) == 65 and raw[0] == 0x04
    assert "=" not in key


@pytest.mark.parametrize(
    "ua, label",
    [
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
            "iPhone",
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Mac · Chrome",
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/18.0 Safari/605.1.15",
            "Mac · Safari",
        ),
        (None, "기기"),
    ],
)
def test_device_label(ua, label):
    assert push.device_label(ua) == label


@pytest.mark.parametrize(
    "scheme, host, subject",
    [
        ("https", "mac.example.ts.net", "https://mac.example.ts.net"),
        ("HTTPS", "Mac.Example.ts.net:8443", "https://mac.example.ts.net:8443"),
        ("http", "127.0.0.1:8765", push.FALLBACK_SUBJECT),
        ("https", "evil.example/path", push.FALLBACK_SUBJECT),
        ("https", None, push.FALLBACK_SUBJECT),
    ],
)
def test_subject_origin(scheme, host, subject):
    assert push.subject_origin(scheme, host) == subject


def test_subjects_pass_py_vapid_validation():
    from py_vapid import _check_sub

    assert _check_sub(push.FALLBACK_SUBJECT)
    assert _check_sub(push.subject_origin("https", "mac.example.ts.net"))


# ---- send outcome mapping ---------------------------------------------------


def _fake_webpush(monkeypatch, *, status=None, exc=None):
    import pywebpush

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        if status is not None:
            raise pywebpush.WebPushException("push failed", response=SimpleNamespace(status_code=status))
        if exc is not None:
            raise exc
        return SimpleNamespace(status_code=201)

    monkeypatch.setattr(pywebpush, "webpush", fake)
    return calls


async def _send(message=None):
    target = push.PushTarget(endpoint=ENDPOINT, subject="https://mac.example.ts.net", **KEYS)
    return await push.send_push(target, message or {"title": "t"}, push.load_or_create_vapid())


@pytest.mark.asyncio
async def test_send_ok_carries_payload_ttl_and_subject(monkeypatch):
    calls = _fake_webpush(monkeypatch)
    outcome = await _send({"title": "오늘의 관측 2", "body": "A 외 1건"})
    assert outcome.status == "ok"
    (call,) = calls
    assert json.loads(call["data"]) == {"title": "오늘의 관측 2", "body": "A 외 1건"}
    assert call["ttl"] == push.PUSH_TTL_SECONDS
    assert call["vapid_claims"]["sub"] == "https://mac.example.ts.net"
    assert call["subscription_info"]["endpoint"] == ENDPOINT


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 410])
async def test_send_gone_subscription_is_expired(monkeypatch, status):
    _fake_webpush(monkeypatch, status=status)
    assert (await _send()).status == "expired"


@pytest.mark.asyncio
async def test_send_other_errors_are_failures(monkeypatch):
    _fake_webpush(monkeypatch, status=500)
    assert (await _send()).status == "failed"
    _fake_webpush(monkeypatch, exc=ConnectionError("offline"))
    outcome = await _send()
    assert outcome.status == "failed" and "ConnectionError" in outcome.detail


# ---- digest wording ---------------------------------------------------------


def test_digest_message_is_event_level():
    digest = push_briefing.BriefingDigest(event_count=7, top_title="Anthropic, 'Claude Duet' 공개")
    msg = digest.message()
    assert msg["title"] == "오늘의 관측 7"
    assert msg["body"] == "Anthropic, 'Claude Duet' 공개 외 6건"
    assert msg["url"] == "/feed"


def test_digest_single_event_has_no_suffix():
    assert push_briefing.BriefingDigest(event_count=1, top_title="X").body == "X"


# ---- routes -----------------------------------------------------------------


def test_push_key_route_returns_public_key(client):
    resp = client.get("/push/key")
    assert resp.status_code == 200
    assert resp.json()["publicKey"] == push.public_key_b64(push.load_or_create_vapid())
    assert resp.headers["cache-control"] == "no-store"


def test_push_key_follows_key_file_rotation(client, _key_file):
    first = client.get("/push/key").json()["publicKey"]
    assert client.get("/push/key").json()["publicKey"] == first  # cached

    _key_file.unlink()  # rotation: delete → regenerated on next use
    second = client.get("/push/key").json()["publicKey"]
    assert second != first
    # The briefing process reads the same file, so both sides agree.
    assert second == push.public_key_b64(push.load_or_create_vapid())


def test_subscribe_upserts_with_device_label(client, monkeypatch):
    seen = {}

    async def fake_upsert(session, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(push_briefing, "upsert_subscription", fake_upsert)
    resp = client.post(
        "/push/subscribe",
        json={"endpoint": ENDPOINT, "keys": KEYS},
        headers={
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)",
            # What `tailscale serve` forwards in front of 127.0.0.1:8765.
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "mac.example.ts.net",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"subscribed": True, "label": "iPhone"}
    assert seen == {
        "endpoint": ENDPOINT,
        "p256dh": "BPpub",
        "auth": "secret",
        "label": "iPhone",
        "origin": "https://mac.example.ts.net",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"endpoint": "http://insecure.example/x", "keys": KEYS},
        {"endpoint": ENDPOINT},
        {"endpoint": ENDPOINT, "keys": {"p256dh": "", "auth": "a"}},
        {"endpoint": "https://x/" + "a" * 3000, "keys": KEYS},
        ["not", "an", "object"],
    ],
)
def test_subscribe_rejects_malformed_bodies(client, monkeypatch, body):
    async def boom(session, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("should not store")

    monkeypatch.setattr(push_briefing, "upsert_subscription", boom)
    assert client.post("/push/subscribe", json=body).status_code == 400


def test_unsubscribe_and_status(client, monkeypatch):
    store = {ENDPOINT}

    async def fake_remove(session, *, endpoint):
        store.discard(endpoint)
        return True

    async def fake_has(session, *, endpoint):
        return endpoint in store

    monkeypatch.setattr(push_briefing, "remove_subscription", fake_remove)
    monkeypatch.setattr(push_briefing, "has_subscription", fake_has)

    assert client.post("/push/status", json={"endpoint": ENDPOINT}).json() == {"subscribed": True}
    assert client.post("/push/unsubscribe", json={"endpoint": ENDPOINT}).json() == {"subscribed": False}
    assert client.post("/push/status", json={"endpoint": ENDPOINT}).json() == {"subscribed": False}


@pytest.mark.real_security
def test_push_routes_keep_login_and_csrf(_web_security_policy):
    client = TestClient(build_web_app())
    # Not logged in → 401, never the key.
    assert client.get("/push/key").status_code == 401
    # Cross-site POST → 403 before auth is even considered.
    resp = client.post(
        "/push/subscribe",
        json={"endpoint": ENDPOINT, "keys": KEYS},
        headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert resp.status_code == 403


def test_settings_page_has_push_toggle(tmp_path, monkeypatch):
    from argos import config_store

    monkeypatch.setattr(config_store, "default_config_path", lambda: tmp_path / "config.toml")
    body = TestClient(build_web_app()).get("/settings").text
    assert 'id="push-settings"' in body
    assert "아침 브리핑 알림" in body
    assert "/static/js/push-settings.js" in body


def test_service_worker_shows_notification_and_opens_feed(web_client):
    body = web_client.get("/sw.js").text
    assert "addEventListener('push'" in body
    assert "showNotification" in body
    assert "addEventListener('notificationclick'" in body
    assert "openWindow" in body and "'/feed'" in body


def test_push_settings_script_is_served(web_client):
    resp = web_client.get("/static/js/push-settings.js")
    assert resp.status_code == 200
    assert "pushManager.subscribe" in resp.text
    assert "/push/subscribe" in resp.text
