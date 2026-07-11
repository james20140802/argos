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

import logging
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
from argos.web.services.feed import count_new_since, encode_cursor, fetch_feed
from argos.web.services.portfolio import fetch_portfolio
from argos.web.services.settings import (
    EDITABLE_FIELDS,
    apply_settings,
    load_settings_view,
)
from argos.web.services.timeline import ReplaceSuccessor, fetch_timeline, replace_successors

_PACKAGE_DIR = Path(__file__).parent  # noqa: E402 — module-level lazy shims below
_log = logging.getLogger("argos.web")


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
    summary, trust_score, source_url, asset_id) or None if the tech_item does
    not exist.
    ``asset_id`` (the user_asset row id, ARG-184) is only used by the detail
    page's action bar — to build the /assets/{id}/untrack URL — and is
    ignored by the feed-card partial.
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
        "trust_score": getattr(tech_item, "trust_score", None),
        "source_url": tech_item.source_url,
        "asset_id": user_asset.id if user_asset else None,
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


async def _is_replace_successor(
    session, predecessor_tech_id: uuid.UUID, successor_tech_id: uuid.UUID
) -> bool:
    """True iff ``successor_tech_id`` is a ``Replace`` successor of
    ``predecessor_tech_id`` in ``tech_succession``.

    The handoff endpoint verifies this before transitioning so a modified or
    stale ``successor_tech_id`` — the lineage changed after the banner was
    rendered, or a hand-crafted POST — cannot archive the predecessor and Keep
    an unrelated (or self) tech item, which would corrupt the portfolio. Only
    ``Replace`` counts: Enhance/Fork are "also Keep" relations, not handoffs.
    Lazy DB import keeps the module import graph DB-free (see the no-DB guard).
    """
    from sqlalchemy import select

    from argos.models.tech_succession import RelationType, TechSuccession

    row = (
        await session.execute(
            select(TechSuccession.id).where(
                TechSuccession.predecessor_id == predecessor_tech_id,
                TechSuccession.successor_id == successor_tech_id,
                TechSuccession.relation_type == RelationType.REPLACE,
            )
        )
    ).first()
    return row is not None


def build_web_app(config_path: Optional[Path] = None) -> FastAPI:
    """Build and return the Argos FastAPI app.

    The app mounts ``/static`` from ``src/argos/web/static/`` and stores
    a configured Jinja2 templates environment on ``app.state.templates``
    so request handlers added by later issues can render views.

    ``config_path`` is the active ``config.toml`` the settings page reads and
    writes. ``_cmd_web`` passes the ``--config``-resolved path so the web UI
    edits the same file the running daemon / scheduled jobs use; when ``None``
    the settings service falls back to ``config_store.default_config_path()``.
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

    def _is_favicon(url: str | None) -> bool:
        """Render-time helper: True when a cover URL is a bare favicon.

        Shares ``argos.crawler._og_image.is_favicon_url`` with the backfill so a
        cache-busting query string (``/favicon.ico?v=2``) still gets the
        favicon-chip branch instead of being stretched as a full cover image.

        The import stays lazy on purpose: ``from argos.crawler._og_image import
        …`` executes ``argos.crawler.__init__``, which transitively pulls in
        ``argos.database``. Hoisting it to registration time would break the
        ``build_web_app`` import-graph isolation invariant
        (``test_build_web_app_does_not_import_argos_database``). Python's import
        cache makes the per-render cost negligible.
        """
        from argos.crawler._og_image import is_favicon_url

        return is_favicon_url(url)

    app.state.templates.env.filters["is_favicon"] = _is_favicon

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
        # Negative deltas (clock skew / future timestamps) fall here too.
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
        # feed-poll.js (ARG-203) reads #feed-list[data-latest-cursor] to know
        # what to poll "newer than". Only meaningful on the genuine first page
        # — a mid-feed "더 보기" fragment or a direct cursor hit has no single
        # "latest" position to anchor polling on, so it's left unset there.
        latest_cursor = (
            encode_cursor(page.items[0].sort_at, page.items[0].id)
            if first_page and page.items
            else ""
        )
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
                "latest_cursor": latest_cursor,
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

    @app.get("/feed/poll")
    async def feed_poll(
        request: Request,
        cursor: Optional[str] = None,
        category: Optional[str] = None,
        session=Depends(_get_session),
    ) -> JSONResponse:
        # ``cursor`` is declared Optional (rather than a bare required ``str``)
        # so a missing value gets the same controlled 400 as a malformed one,
        # instead of FastAPI's default 422 — the AC only specifies "invalid
        # cursor → 400", and this keeps both failure modes on one status code.
        normalized = _normalize_category(category)
        if cursor is None:
            raise HTTPException(status_code=400, detail="invalid feed cursor")
        try:
            new_count = await count_new_since(session, category=normalized, cursor=cursor)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid feed cursor") from exc
        return JSONResponse({"new_count": new_count})

    async def _load_handoff_banners(session, view):
        """Per-asset Replace-successor lookup for the portfolio card's
        handoff banner (ARG-209).

        Bounded by ``lineage_count > 0`` — the aggregate ``fetch_portfolio``
        already computes — so a Keep-only portfolio with no succession links
        at all issues zero extra queries. Keyed by ``user_asset.id`` (the
        card's DOM id), value is the first Replace successor found.
        """
        banners: dict[uuid.UUID, ReplaceSuccessor] = {}
        for asset in (*view.active, *view.quiet):
            if asset.lineage_count <= 0:
                continue
            successors = await replace_successors(session, asset.tech_id)
            if successors:
                banners[asset.id] = successors[0]
        return banners

    async def _render_portfolio(
        request: Request,
        template_name: str,
        category: Optional[str],
        sort: Optional[str],
        cursor: Optional[str],
        session,
    ) -> HTMLResponse:
        normalized_category = _normalize_category(category)
        normalized_sort = sort if sort in _VALID_SORTS else "recency"
        try:
            view = await fetch_portfolio(
                session,
                category=normalized_category,
                sort=normalized_sort,
                cursor=cursor,
            )
        except ValueError as exc:
            # ``cursor`` is user-controlled query state; a stale/corrupted
            # load-more URL must not 500. Translate it to a controlled 400.
            raise HTTPException(
                status_code=400, detail="invalid portfolio query"
            ) from exc
        handoff_banners = await _load_handoff_banners(session, view)
        return request.app.state.templates.TemplateResponse(
            request,
            template_name,
            {
                "view": view,
                "category": normalized_category,
                "sort": normalized_sort,
                "handoff_banners": handoff_banners,
            },
        )

    @app.get("/portfolio", response_class=HTMLResponse)
    async def portfolio(
        request: Request,
        category: Optional[str] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        return await _render_portfolio(
            request, "portfolio.html", category, sort, cursor, session
        )

    @app.get("/portfolio/items", response_class=HTMLResponse)
    async def portfolio_items(
        request: Request,
        category: Optional[str] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        return await _render_portfolio(
            request, "_portfolio_items.html", category, sort, cursor, session
        )

    @app.get("/portfolio/{asset_id}/timeline", response_class=HTMLResponse)
    async def portfolio_asset_timeline(
        request: Request,
        asset_id: str,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        # ``asset_id`` is user-controlled path state; a malformed UUID or an
        # asset that no longer exists must not 500 — both render the same
        # controlled 404 fragment (this endpoint is only ever hit via HTMX,
        # never as a full-page navigation).
        try:
            parsed_id = uuid.UUID(asset_id)
        except ValueError:
            return _error_fragment(request, 404, "not found")

        tech_id = await _resolve_user_asset_tech_id(session, parsed_id)
        if tech_id is None:
            return _error_fragment(request, 404, "not found")

        events = await fetch_timeline(session, tech_id, limit=5)
        return request.app.state.templates.TemplateResponse(
            request, "_timeline.html", {"events": events}
        )

    def _render_not_found(request: Request) -> HTMLResponse:
        return request.app.state.templates.TemplateResponse(
            request, "not_found.html", {}, status_code=404
        )

    def _render_error(request: Request, request_id: str) -> HTMLResponse:
        return request.app.state.templates.TemplateResponse(
            request, "error.html", {"request_id": request_id}, status_code=500
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> HTMLResponse:
        # Unhandled exceptions only — HTTPException (404/400/...) is routed
        # through Starlette's separate HTTPException middleware and never
        # reaches this handler, so the themed 404 page above is unaffected.
        # The stacktrace goes to the log only; the response body carries
        # nothing but a short request id.
        request_id = uuid.uuid4().hex[:8]
        _log.exception("unhandled error [%s] on %s", request_id, request.url.path)
        return _render_error(request, request_id)

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
        partial_name: str = "_feed_card.html",
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
        # absence, after a toggle-off). A None here means the TechItem row was
        # deleted between the guard above and this reload — return the 404
        # fragment rather than let SimpleNamespace(**None) raise a 500.
        item = await _load_feed_card_context(session, parsed_id)
        if item is None:
            return _error_fragment(request, 404, "not found")
        return _action_response(
            request, item, partial_name, is_featured=is_featured
        )

    # ``context=detail`` (ARG-184) tells keep/pass/untrack to re-render the
    # item-detail page's standalone action bar (``_detail_actions.html``)
    # instead of a feed-card fragment. It is opt-in via a query param so the
    # feed's existing hx-post calls (which never send it) are byte-for-byte
    # unaffected — the default keeps returning ``_feed_card.html``.
    def _partial_for(context: str) -> str:
        return "_detail_actions.html" if context == "detail" else "_feed_card.html"

    @app.post("/items/{item_id}/keep", response_class=HTMLResponse)
    async def keep_item(
        request: Request,
        item_id: str,
        featured: bool = False,
        active: bool = False,
        context: str = "feed",
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
            partial_name=_partial_for(context),
        )

    @app.post("/items/{item_id}/pass", response_class=HTMLResponse)
    async def pass_item(
        request: Request,
        item_id: str,
        featured: bool = False,
        active: bool = False,
        context: str = "feed",
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
            partial_name=_partial_for(context),
        )

    @app.post("/assets/{user_asset_id}/untrack", response_class=HTMLResponse)
    async def untrack_asset(
        request: Request,
        user_asset_id: str,
        context: str = "feed",
        tech_id: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        from argos.models.user_asset import AssetStatus
        from argos.slack.services.asset_transition import TransitionOutcome

        try:
            parsed_id = uuid.UUID(user_asset_id)
        except ValueError:
            return _error_fragment(request, 404, "not found")

        resolved_tech_id = await _resolve_user_asset_tech_id(session, parsed_id)
        if resolved_tech_id is not None:
            outcome = await transition_asset(
                session, resolved_tech_id, AssetStatus.ARCHIVED
            )
            if outcome is not TransitionOutcome.NOOP:
                # Both a real Keep→Archived transition (TRANSITIONED) and a
                # freshly-inserted Archived row (CREATED — the UserAsset was
                # concurrently cleared between resolve and here) mutate state and
                # must be persisted; the request session does not auto-commit
                # (see keep/pass above). Only NOOP (already Archived) changed
                # nothing, so it needs no commit.
                await session.commit()

        if context == "detail":
            # The detail page has exactly one card on screen, so untracking
            # can't just delete it like the portfolio does — it re-renders the
            # action bar in place instead (ARG-184). ``tech_id`` is threaded
            # through as a query param because a stale/already-cleared
            # ``user_asset_id`` cannot be resolved back to it after the fact.
            detail_tech_id = resolved_tech_id
            if detail_tech_id is None and tech_id is not None:
                try:
                    detail_tech_id = uuid.UUID(tech_id)
                except ValueError:
                    return _error_fragment(request, 404, "not found")
            if detail_tech_id is None:
                return _error_fragment(request, 404, "not found")
            item = await _load_feed_card_context(session, detail_tech_id)
            if item is None:
                return _error_fragment(request, 404, "not found")
            return _action_response(request, item, "_detail_actions.html")

        # Untracking archives the asset, dropping it out of the Keep-only
        # portfolio. A missing row (a stale cached /portfolio card whose asset
        # was already cleared — e.g. a feed toggle-off deleted the UserAsset) or
        # an already-Archived row both mean the desired end state already holds.
        # Return an empty body so the HTMX ``outerHTML`` swap removes the card,
        # idempotently — a 404/409 error fragment would leave a stale, dead card
        # displaying an error even though the untrack goal is satisfied.
        return HTMLResponse("", status_code=200)

    @app.post("/assets/{user_asset_id}/handoff", response_class=HTMLResponse)
    async def handoff_asset(
        request: Request,
        user_asset_id: str,
        successor_tech_id: Optional[str] = None,
        context: Optional[str] = None,
        session=Depends(_get_session),
    ) -> HTMLResponse:
        """Succession handoff (ARG-209): archive the predecessor asset (named
        by ``user_asset_id``), Keep the successor (``successor_tech_id``).

        Both transitions reuse ``transition_asset`` — each independently
        upserted/logged to ``track_history`` — so the action is idempotent by
        construction: replaying it is two NOOPs. If the successor is already
        Keep (the "이미 tracking 중" case), only the predecessor's archive has
        any effect; the successor's transition_asset call still runs but
        NOOPs.

        The successor is verified to be a real ``Replace`` successor of the
        predecessor before anything transitions, so a modified or stale
        ``successor_tech_id`` cannot hand the asset off to an unrelated tech
        item and corrupt the portfolio.

        Response by ``context``: ``detail`` re-renders the whole detail action
        area (``_detail_actions.html``) — the predecessor is now Archived, so
        the banner drops and the bar reflects the new state in place (the
        button targets ``#detail-actions-<id>``). Otherwise an empty 200 body
        mirrors ``untrack``'s portfolio contract: the caller's hx-swap removes
        the whole predecessor card.
        """
        from argos.models.user_asset import AssetStatus

        try:
            parsed_asset_id = uuid.UUID(user_asset_id)
        except ValueError:
            return _error_fragment(request, 404, "not found")

        if successor_tech_id is None:
            return _error_fragment(request, 404, "not found")
        try:
            parsed_successor_id = uuid.UUID(successor_tech_id)
        except ValueError:
            return _error_fragment(request, 404, "not found")

        predecessor_tech_id = await _resolve_user_asset_tech_id(
            session, parsed_asset_id
        )
        if predecessor_tech_id is None:
            return _error_fragment(request, 404, "not found")

        # Reject a handoff whose successor is not an actual Replace successor of
        # this predecessor (stale banner after the lineage changed, or a
        # hand-crafted POST). Without this a single request could archive the
        # asset and Keep an arbitrary/self UUID, corrupting the portfolio.
        if not await _is_replace_successor(
            session, predecessor_tech_id, parsed_successor_id
        ):
            return _error_fragment(request, 409, "invalid successor")

        await transition_asset(session, predecessor_tech_id, AssetStatus.ARCHIVED)
        await transition_asset(session, parsed_successor_id, AssetStatus.KEEP)
        await session.commit()

        if context == "detail":
            # Re-render the predecessor's detail action area, now Archived: the
            # handoff banner is gone and the bar shows Keep/Pass instead of a
            # stale Untrack for a state that no longer exists.
            item = await _load_feed_card_context(session, predecessor_tech_id)
            if item is None:
                return _error_fragment(request, 404, "not found")
            return _action_response(request, item, "_detail_actions.html")

        return HTMLResponse("", status_code=200)

    # ---- Settings (ARG-186) ------------------------------------------------
    # No DB session: settings read/write only ``config.toml`` via config_store.
    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        saved = request.query_params.get("saved") == "1"
        view = load_settings_view(config_path, saved=saved)
        return request.app.state.templates.TemplateResponse(
            request, "settings.html", {"view": view}
        )

    @app.post("/settings")
    async def save_settings(request: Request) -> Response:
        form = await request.form()
        updates: dict[str, str] = {}
        for spec in EDITABLE_FIELDS:
            if spec.kind == "bool":
                # A checkbox posts its value only when checked, so absence alone
                # is ambiguous: an intentional uncheck vs. a partial POST that
                # never carried the field. The template emits a hidden
                # ``<key>__present`` marker next to every checkbox, so we only
                # treat absence as an uncheck when that marker proves the field
                # was actually on this form. A partial/non-browser POST that
                # omits the marker leaves the bool untouched.
                if f"{spec.key}__present" in form:
                    updates[spec.key] = "true" if spec.key in form else "false"
            elif spec.kind == "weekdays":
                # A toggle-button group posts one entry per checked day (and
                # nothing when all are off). Like the bool checkbox it carries a
                # hidden ``<key>__present`` marker so an all-off submission is a
                # real (validation-rejected) empty list, not a partial POST.
                if f"{spec.key}__present" in form:
                    updates[spec.key] = ",".join(form.getlist(spec.key))
            elif spec.key in form:
                # Only update non-bool fields the form actually carried. The full
                # settings form submits every input (empty strings included), so
                # this is a no-op for a real browser but keeps a partial POST from
                # blanking untouched fields (e.g. briefing.weekdays min_length=1).
                updates[spec.key] = str(form[spec.key])

        errors = apply_settings(updates, config_path)
        if errors:
            # Post-Redirect-Get is skipped on failure: re-render in place so the
            # user keeps their typed values and sees inline field errors.
            view = load_settings_view(config_path, submitted=updates, errors=errors)
            return request.app.state.templates.TemplateResponse(
                request, "settings.html", {"view": view}, status_code=400
            )
        # PRG: redirect so a refresh doesn't re-POST the form.
        return RedirectResponse("/settings?saved=1", status_code=303)

    return app
