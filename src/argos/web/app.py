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
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from argos.web.services.feed import fetch_feed

_PACKAGE_DIR = Path(__file__).parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"

_VALID_CATEGORIES = ("Mainstream", "Alpha")


async def _get_session():
    """Per-request async DB session.

    Imports ``argos.database`` lazily inside the call so that merely
    constructing the app (``build_web_app``) never pulls the DB engine into
    the import graph — release.yml CI runs pytest without Postgres and the
    ``test_build_web_app_does_not_import_argos_database`` guard enforces this.
    """
    from argos.database import get_session as _db_get_session

    async for session in _db_get_session():
        yield session


def _normalize_category(category: Optional[str]) -> Optional[str]:
    """Coerce an arbitrary ?category= value to a valid filter or None (전체)."""
    return category if category in _VALID_CATEGORIES else None


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

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/feed")

    @app.get("/feed", response_class=HTMLResponse)
    async def feed(
        request: Request,
        category: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        normalized = _normalize_category(category)
        page = await fetch_feed(session, category=normalized, cursor=cursor)
        return request.app.state.templates.TemplateResponse(
            request,
            "feed.html",
            {
                "items": page.items,
                "next_cursor": page.next_cursor,
                "category": normalized,
            },
        )

    @app.get("/feed/items", response_class=HTMLResponse)
    async def feed_items(
        request: Request,
        category: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        normalized = _normalize_category(category)
        page = await fetch_feed(session, category=normalized, cursor=cursor)
        return request.app.state.templates.TemplateResponse(
            request,
            "_feed_items.html",
            {
                "items": page.items,
                "next_cursor": page.next_cursor,
                "category": normalized,
            },
        )

    return app
