from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import argos.web
from argos.web.app import _get_session, build_web_app

PKG = Path(argos.web.__file__).parent


def _client_that_500s(monkeypatch) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None
    app.dependency_overrides[_get_session] = _fake_session

    async def _boom(session, *, category=None, cursor=None, limit=20):
        raise RuntimeError("kaboom stacktrace secret")
    monkeypatch.setattr("argos.web.app.fetch_feed", _boom)

    async def _fake_activity(session, limit=12):
        return []
    monkeypatch.setattr("argos.web.app.fetch_activity", _fake_activity)
    return TestClient(app, raise_server_exceptions=False)


def test_unhandled_exception_returns_500(monkeypatch):
    resp = _client_that_500s(monkeypatch).get("/feed")
    assert resp.status_code == 500


def test_500_page_is_themed(monkeypatch):
    body = _client_that_500s(monkeypatch).get("/feed").text
    # extends base.html -> masthead/rail wordmark present
    assert "ARGOS" in body
    assert "500" in body
    assert "/feed" in body  # link back


def test_500_page_hides_stacktrace(monkeypatch):
    body = _client_that_500s(monkeypatch).get("/feed").text
    assert "kaboom stacktrace secret" not in body
    assert "Traceback" not in body


def test_error_template_extends_base():
    html = (PKG / "templates" / "error.html").read_text(encoding="utf-8")
    assert 'extends "base.html"' in html


def test_404_still_renders_not_found(monkeypatch):
    # regression: the Exception handler must not swallow HTTPException(404)
    app = build_web_app()
    async def _fake_session():
        yield None
    app.dependency_overrides[_get_session] = _fake_session
    resp = TestClient(app, raise_server_exceptions=False).get("/item/not-a-uuid")
    assert resp.status_code == 404
    assert "404" in resp.text
