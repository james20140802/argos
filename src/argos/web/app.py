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

import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from argos.web.services.detail import fetch_item_detail
from argos.web.services.feed import fetch_feed
from argos.web.services.portfolio import fetch_portfolio

_PACKAGE_DIR = Path(__file__).parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"

_VALID_CATEGORIES = ("Mainstream", "Alpha")
_VALID_SORTS = ("recency", "trust")


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

    async def _render_feed(
        request: Request,
        template_name: str,
        category: Optional[str],
        cursor: Optional[str],
        session,
    ) -> HTMLResponse:
        normalized = _normalize_category(category)
        try:
            page = await fetch_feed(session, category=normalized, cursor=cursor)
        except ValueError as exc:
            # ``cursor`` is user-controlled query state; a stale/corrupted
            # load-more URL must not 500. Translate it to a controlled 400.
            raise HTTPException(status_code=400, detail="invalid feed cursor") from exc
        return request.app.state.templates.TemplateResponse(
            request,
            template_name,
            {
                "items": page.items,
                "next_cursor": page.next_cursor,
                "category": normalized,
            },
        )

    @app.get("/feed", response_class=HTMLResponse)
    async def feed(
        request: Request,
        category: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        return await _render_feed(request, "feed.html", category, cursor, session)

    @app.get("/feed/items", response_class=HTMLResponse)
    async def feed_items(
        request: Request,
        category: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        return await _render_feed(
            request, "_feed_items.html", category, cursor, session
        )

    async def _render_portfolio(
        request: Request,
        category: Optional[str],
        sort: Optional[str],
        session,
    ) -> HTMLResponse:
        normalized_category = _normalize_category(category)
        normalized_sort = sort if sort in _VALID_SORTS else "recency"
        try:
            view = await fetch_portfolio(
                session, category=normalized_category, sort=normalized_sort
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="invalid portfolio query"
            ) from exc
        return request.app.state.templates.TemplateResponse(
            request,
            "portfolio.html",
            {
                "view": view,
                "category": normalized_category,
                "sort": normalized_sort,
            },
        )

    @app.get("/portfolio", response_class=HTMLResponse)
    async def portfolio(
        request: Request,
        category: Optional[str] = None,
        sort: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        return await _render_portfolio(request, category, sort, session)

    def _render_not_found(request: Request) -> HTMLResponse:
        return request.app.state.templates.TemplateResponse(
            request, "not_found.html", {}, status_code=404
        )

    @app.get("/item/{item_id}", response_class=HTMLResponse)
    async def item_detail(
        request: Request,
        item_id: str,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        # ``item_id`` is user-controlled path state; a malformed UUID must not
        # 500. Translate it to the same 404 page as a real miss.
        try:
            parsed_id = uuid.UUID(item_id)
        except ValueError:
            return _render_not_found(request)

        item = await fetch_item_detail(session, parsed_id)
        if item is None:
            return _render_not_found(request)

        return request.app.state.templates.TemplateResponse(
            request, "item_detail.html", {"item": item}
        )

    return app
