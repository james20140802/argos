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
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from argos.web.services.activity import fetch_activity
from argos.web.services.detail import fetch_item_detail
from argos.web.services.feed import fetch_feed
from argos.web.services.portfolio import fetch_portfolio

_PACKAGE_DIR = Path(__file__).parent  # noqa: E402 — module-level lazy shims below


async def transition_asset(session, tech_id: uuid.UUID, target_status):
    """Lazy shim — delegates to argos.slack.services.asset_transition.

    Defined at module level so tests can monkeypatch ``argos.web.app.transition_asset``
    without triggering an eager ``argos.database`` import at app-construction time.
    """
    from argos.slack.services.asset_transition import (
        transition_asset as _real_transition_asset,
    )

    return await _real_transition_asset(session, tech_id, target_status)


async def toggle_asset(
    session, tech_id: uuid.UUID, target_status, *, currently_active: bool = False
):
    """Lazy shim — delegates to argos.slack.services.asset_transition.toggle_asset.

    Kept at module level (like ``transition_asset``) so tests can monkeypatch
    ``argos.web.app.toggle_asset`` without an eager ``argos.database`` import.
    """
    from argos.slack.services.asset_transition import toggle_asset as _real_toggle_asset

    return await _real_toggle_asset(
        session, tech_id, target_status, currently_active=currently_active
    )
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"
_ASSETS_DIR = _PACKAGE_DIR / "assets"

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


async def _load_feed_card_context(session, tech_id: uuid.UUID):
    """Fetch the minimal shape the feed-card partial needs after a transition.

    Returns a mapping with keys (id, title, status, category, image_url,
    summary, source_url) or None if the tech_item does not exist.
    """
    from sqlalchemy import select

    from argos.models.tech_item import TechItem
    from argos.models.user_asset import UserAsset

    row = (
        await session.execute(
            select(TechItem, UserAsset)
            .join(UserAsset, UserAsset.tech_id == TechItem.id, isouter=True)
            .where(TechItem.id == tech_id)
        )
    ).first()
    if row is None:
        return None
    tech_item, user_asset = row
    return {
        "id": tech_item.id,
        "title": tech_item.title,
        "status": user_asset.status if user_asset else None,
        "category": tech_item.category,
        "image_url": getattr(tech_item, "image_url", None),
        "summary": getattr(tech_item, "summary", None),
        "source_url": tech_item.source_url,
    }


async def _resolve_user_asset_tech_id(session, user_asset_id: uuid.UUID):
    """Resolve a user_asset row to its tech_id.

    Returns ``None`` if no row exists.  Lazy DB import keeps the module-level
    import graph free of ``argos.database`` (see the no-DB guard test).
    """
    from sqlalchemy import select

    from argos.models.user_asset import UserAsset

    row = (
        await session.execute(
            select(UserAsset.tech_id).where(UserAsset.id == user_asset_id)
        )
    ).first()
    if row is None:
        return None
    return row[0]


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

    def _domain_of(url: str | None) -> str:
        """Render-time helper: netloc of a URL, '' when unparseable."""
        if not url:
            return ""
        try:
            return urlsplit(url).netloc
        except ValueError:
            return ""

    app.state.templates.env.filters["domain"] = _domain_of

    def _reltime(value) -> str:
        """Render-time helper: a compact Korean relative time for the ticker.

        Display-only; ``datetime.now`` is acceptable here (not on a code path
        that needs deterministic output for tests). Falls back to an ISO date
        for anything older than a week or unparseable.
        """
        from datetime import datetime, timezone

        if not isinstance(value, datetime):
            return ""
        when = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - when
        secs = delta.total_seconds()
        if secs < 0:
            return "방금"
        if secs < 60:
            return "방금"
        if secs < 3600:
            return f"{int(secs // 60)}분 전"
        if secs < 86400:
            return f"{int(secs // 3600)}시간 전"
        if secs < 604800:
            return f"{int(secs // 86400)}일 전"
        return when.strftime("%Y-%m-%d")

    app.state.templates.env.filters["reltime"] = _reltime

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/feed")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest() -> JSONResponse:
        return JSONResponse(
            {
                "name": "ARGOS — Observatory",
                "short_name": "ARGOS",
                "description": (
                    "Local-first AI technology observatory — feed, "
                    "portfolio, and signals."
                ),
                "start_url": "/feed",
                "scope": "/",
                "display": "standalone",
                "orientation": "portrait",
                "theme_color": "#0b0d12",
                "background_color": "#0b0d12",
                "lang": "ko",
                "icons": [
                    {
                        "src": "/static/img/icons/icon-192.png",
                        "sizes": "192x192",
                        "type": "image/png",
                        "purpose": "any",
                    },
                    {
                        "src": "/static/img/icons/icon-512.png",
                        "sizes": "512x512",
                        "type": "image/png",
                        "purpose": "any",
                    },
                    {
                        "src": "/static/img/icons/icon-maskable-512.png",
                        "sizes": "512x512",
                        "type": "image/png",
                        "purpose": "maskable",
                    },
                ],
            },
            media_type="application/manifest+json",
        )

    # Pre-read the SW body once at app construction so /sw.js never does
    # disk I/O on the request path (it's served on every install/update).
    # The file lives outside the /static/ mount on purpose: serving it only
    # from the root-scope route avoids a second copy at /static/sw.js that
    # would have a needlessly narrow scope.
    _sw_body = (_ASSETS_DIR / "sw.js").read_bytes()

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker() -> Response:
        return Response(
            content=_sw_body,
            media_type="application/javascript",
            headers={
                "Service-Worker-Allowed": "/",
                "Cache-Control": "no-cache",
            },
        )

    async def _render_feed(
        request: Request,
        template_name: str,
        category: Optional[str],
        cursor: Optional[str],
        session,
        *,
        first_page: bool,
        include_activity: bool = False,
    ) -> HTMLResponse:
        normalized = _normalize_category(category)
        try:
            page = await fetch_feed(session, category=normalized, cursor=cursor)
        except ValueError as exc:
            # ``cursor`` is user-controlled query state; a stale/corrupted
            # load-more URL must not 500. Translate it to a controlled 400.
            raise HTTPException(status_code=400, detail="invalid feed cursor") from exc
        # The signal ticker is full-page chrome (feed.html), never part of the
        # HTMX "더 보기" fragment — so it's only fetched for the initial render.
        activity = await fetch_activity(session) if include_activity else []
        return request.app.state.templates.TemplateResponse(
            request,
            template_name,
            {
                "items": page.items,
                "next_cursor": page.next_cursor,
                "category": normalized,
                # Featured hero is keyed on first-page index 0 only; the HTMX
                # "더 보기" fragment (GET /feed/items) must never re-emit a hero
                # mid-scroll, so it renders with first_page=False.
                "first_page": first_page,
                "activity": activity,
            },
        )

    @app.get("/feed", response_class=HTMLResponse)
    async def feed(
        request: Request,
        category: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        # Featured hero belongs to the genuine first page only. A direct hit on
        # ``/feed?cursor=<token>`` (browser history, shared link) is a mid-feed
        # page, so its index-0 item must not be promoted to the hero slot.
        return await _render_feed(
            request, "feed.html", category, cursor, session,
            first_page=cursor is None,
            include_activity=True,
        )

    @app.get("/feed/items", response_class=HTMLResponse)
    async def feed_items(
        request: Request,
        category: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        return await _render_feed(
            request, "_feed_items.html", category, cursor, session, first_page=False
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

    def _error_fragment(request: Request, status_code: int, message: str) -> HTMLResponse:
        return HTMLResponse(
            f'<div class="action-error" data-status="{status_code}">{message}</div>',
            status_code=status_code,
        )

    def _action_response(
        request: Request,
        item: dict,
        partial_name: str,
        *,
        is_featured: bool = False,
    ) -> HTMLResponse:
        from types import SimpleNamespace

        return request.app.state.templates.TemplateResponse(
            request,
            partial_name,
            {"item": SimpleNamespace(**item), "is_featured": is_featured},
        )

    async def _toggle_item(
        request: Request,
        item_id: str,
        target_status,
        session,
        *,
        is_featured: bool,
        currently_active: bool = False,
    ) -> HTMLResponse:
        try:
            parsed_id = uuid.UUID(item_id)
        except ValueError:
            return _error_fragment(request, 404, "not found")

        item = await _load_feed_card_context(session, parsed_id)
        if item is None:
            return _error_fragment(request, 404, "not found")

        # Toggle semantics: ``currently_active`` is the state the *client* drew
        # (the button showed ✓). Deriving set-vs-clear from what the user saw —
        # not the live DB row — keeps a stale service-worker-cached card from
        # inverting the action (see toggle_asset docstring).
        await toggle_asset(
            session, parsed_id, target_status, currently_active=currently_active
        )

        # ``_get_session`` only opens and closes the AsyncSession; it does not
        # auto-commit. Without this explicit commit the change is rolled back
        # when the request dependency closes, so a reload would still see the
        # old status. The Slack handlers commit after the same service call.
        await session.commit()

        # Reload so the re-rendered card reflects the fresh status (or its
        # absence, after a toggle-off).
        item = await _load_feed_card_context(session, parsed_id)
        return _action_response(
            request, item, "_feed_card.html", is_featured=is_featured
        )

    @app.post("/items/{item_id}/keep", response_class=HTMLResponse)
    async def keep_item(
        request: Request,
        item_id: str,
        featured: bool = False,
        active: bool = False,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        from argos.models.user_asset import AssetStatus

        return await _toggle_item(
            request,
            item_id,
            AssetStatus.KEEP,
            session,
            is_featured=featured,
            currently_active=active,
        )

    @app.post("/items/{item_id}/pass", response_class=HTMLResponse)
    async def pass_item(
        request: Request,
        item_id: str,
        featured: bool = False,
        active: bool = False,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        from argos.models.user_asset import AssetStatus

        return await _toggle_item(
            request,
            item_id,
            AssetStatus.ARCHIVED,
            session,
            is_featured=featured,
            currently_active=active,
        )

    @app.post("/assets/{user_asset_id}/untrack", response_class=HTMLResponse)
    async def untrack_asset(
        request: Request,
        user_asset_id: str,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        from argos.models.user_asset import AssetStatus
        from argos.slack.services.asset_transition import TransitionOutcome

        try:
            parsed_id = uuid.UUID(user_asset_id)
        except ValueError:
            return _error_fragment(request, 404, "not found")

        tech_id = await _resolve_user_asset_tech_id(session, parsed_id)
        if tech_id is None:
            return _error_fragment(request, 404, "not found")

        outcome = await transition_asset(session, tech_id, AssetStatus.ARCHIVED)
        if outcome is TransitionOutcome.NOOP:
            return _error_fragment(request, 409, "already archived")

        # See keep/pass above: the request session does not auto-commit.
        await session.commit()

        # Untracking archives the asset, so it drops out of the Keep-only
        # portfolio. Return an empty body so the HTMX ``outerHTML`` swap removes
        # the card from the page rather than leaving a stale entry behind.
        return HTMLResponse("", status_code=200)

    return app
