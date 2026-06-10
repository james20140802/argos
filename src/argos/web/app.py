"""FastAPI app factory for the Argos web layer (ARG-133).

Mirrors the structure of `argos.slack.app`: a single `build_web_app()`
factory that registers routes, mounts static assets, and wires the
Jinja2 template environment. The factory deliberately does NOT open a
DB connection at construction time — the shared async engine in
`argos.database` is lazy and is only touched by request handlers that
need it. This keeps tests (and release.yml CI, which has no Postgres)
runnable without a live database.

ARG-134 onward will add routes, templates, and static assets; this
foundation only ships /healthz so deployments can probe liveness.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_PACKAGE_DIR = Path(__file__).parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"


def build_web_app() -> FastAPI:
    """Build and return the Argos FastAPI app.

    The app mounts ``/static`` from ``src/argos/web/static/`` and stores
    a configured Jinja2 templates environment on ``app.state.templates``
    so request handlers added by later issues can render views.
    """
    app = FastAPI(
        title="Argos Web",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.mount(
        "/static",
        StaticFiles(directory=_STATIC_DIR, check_dir=False),
        name="static",
    )
    app.state.templates = Jinja2Templates(directory=_TEMPLATES_DIR)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
