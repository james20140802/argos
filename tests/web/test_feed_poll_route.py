from __future__ import annotations
import uuid
from datetime import datetime, timezone
from starlette.testclient import TestClient
from argos.web.app import _get_session, build_web_app
from argos.web.services.feed import encode_cursor


def _client(monkeypatch, *, new_count=0, raise_value_error=False, capture=None):
    app = build_web_app()
    async def _fake_session():
        yield None
    app.dependency_overrides[_get_session] = _fake_session
    async def _fake_count(session, *, category=None, cursor):
        if capture is not None:
            capture.append({"category": category, "cursor": cursor})
        if raise_value_error:
            raise ValueError("bad cursor")
        return new_count
    monkeypatch.setattr("argos.web.app.count_new_since", _fake_count)
    return TestClient(app)


def test_poll_returns_new_count(monkeypatch):
    c = _client(monkeypatch, new_count=3)
    cur = encode_cursor(datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc), uuid.uuid4())
    resp = c.get(f"/feed/poll?cursor={cur}")
    assert resp.status_code == 200
    assert resp.json() == {"new_count": 3}


def test_poll_invalid_cursor_is_400(monkeypatch):
    c = _client(monkeypatch, raise_value_error=True)
    resp = c.get("/feed/poll?cursor=garbage")
    assert resp.status_code == 400


def test_poll_passes_category(monkeypatch):
    cap = []
    c = _client(monkeypatch, new_count=0, capture=cap)
    cur = encode_cursor(datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc), uuid.uuid4())
    c.get(f"/feed/poll?cursor={cur}&category=Alpha")
    assert cap and cap[0]["category"] == "Alpha"
