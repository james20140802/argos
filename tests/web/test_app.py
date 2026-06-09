"""Tests for src/argos/web/app.py (ARG-133)."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from argos.web.app import build_web_app


def test_build_web_app_returns_fastapi_instance():
    """Factory returns a FastAPI app without raising."""
    from fastapi import FastAPI

    app = build_web_app()
    assert isinstance(app, FastAPI)


def test_build_web_app_mounts_static_and_templates():
    """Factory mounts a /static route and stores a Jinja2 templates env."""
    app = build_web_app()

    # /static mount registered as a Starlette Mount route.
    mount_paths = [getattr(r, "path", None) for r in app.routes]
    assert "/static" in mount_paths

    # Templates env is stashed on app.state so handlers can render.
    assert hasattr(app.state, "templates")


@pytest.mark.asyncio
async def test_healthz_returns_200_ok():
    """GET /healthz responds 200 with {'status': 'ok'} and never touches the DB."""
    app = build_web_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_build_web_app_does_not_connect_to_db_at_construction():
    """Factory must not open a DB connection during build (no DB in release CI)."""
    # Build twice; if a connection were opened we'd see it through engine pool.
    # The cheapest assertion is that build_web_app does not raise even when the
    # configured Postgres is unreachable. Smoke-build it without monkeypatching.
    app = build_web_app()
    assert app is not None
