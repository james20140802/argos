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


def test_build_web_app_does_not_import_argos_database():
    """build_web_app must not pull argos.database into the import graph.

    Enforced via a fresh subprocess so sys.modules isn't polluted by other
    tests in the session. release.yml runs pytest without Postgres — any
    DB-touching import at construction would break that CI.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from argos.web.app import build_web_app\n"
                "build_web_app()\n"
                "assert 'argos.database' not in sys.modules, "
                "'argos.database leaked into argos.web.app import graph'\n"
            ),
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
