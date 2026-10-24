"""App-level authentication + CSRF defence for the Argos web app (ARG-195).

Two independent gates, installed as one outermost HTTP middleware:

1. **CSRF** — every state-changing request (POST/PUT/PATCH/DELETE) must prove
   it came from our own pages: ``Sec-Fetch-Site: same-origin`` (all current
   browsers send it), or — when that header is missing — an ``Origin`` whose
   host matches ``Host`` / ``X-Forwarded-Host``. Anything else is 403. No
   template or JS change is needed: HTMX, ``fetch`` and ``sendBeacon`` from our
   own pages all satisfy it.
2. **Authentication** — every non-exempt request needs the ``argos_session``
   cookie (HMAC of the local web token, see :mod:`argos.web.auth`). Without it:

   * full-page HTML GET → **401 with the login form rendered inline** (no
     redirect, so the service worker never caches a login page as /feed);
   * HTMX request → 401 + ``HX-Redirect: /login?next=…``;
   * anything else → 401 JSON.

Identity headers such as ``Tailscale-User-Login`` are deliberately ignored —
they are trivially forgeable by anything that can reach the port.

The policy is looked up **per request** (``app.state.security_policy``, falling
back to :func:`default_policy`) so tests can swap it without rebuilding apps.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from argos.web import auth

_log = logging.getLogger("argos.web.security")

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DEFAULT_NEXT = "/feed"

# Paths reachable without a session: liveness probe, the login flow itself and
# what the browser needs to render it / install the PWA.
_EXEMPT_EXACT = frozenset(
    {"/healthz", "/login", "/logout", "/manifest.webmanifest", "/sw.js", "/favicon.ico"}
)
_EXEMPT_PREFIXES = ("/static/",)


@dataclass(frozen=True)
class SecurityPolicy:
    """Whether auth/CSRF is enforced and where the web token lives."""

    enabled: bool = True
    token_path: Optional[Path] = None
    failed_login_delay: float = 0.5
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def resolved_token_path(self) -> Path:
        return self.token_path or auth.default_token_path()

    def current_token(self) -> Optional[str]:
        """Read the token, cached by file mtime so a rotation applies live."""
        path = self.resolved_token_path()
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        cached = self._cache.get("token")
        if cached and cached[0] == (str(path), mtime):
            return cached[1]
        token = auth.read_token(path)
        self._cache["token"] = ((str(path), mtime), token)
        return token


# Tests swap this (tests/web/conftest.py) — production never touches it.
_default_policy_override: Optional[SecurityPolicy] = None


def default_policy() -> SecurityPolicy:
    if _default_policy_override is not None:
        return _default_policy_override
    return SecurityPolicy()


def _policy_for(request: Request) -> SecurityPolicy:
    policy = getattr(request.app.state, "security_policy", None)
    return policy if policy is not None else default_policy()


def is_exempt(path: str) -> bool:
    return path in _EXEMPT_EXACT or path.startswith(_EXEMPT_PREFIXES)


def safe_next(value: Optional[str]) -> str:
    """Only same-site absolute paths; anything else falls back to /feed."""
    if not value or not value.startswith("/") or value.startswith("//"):
        return DEFAULT_NEXT
    if "\\" in value or any(ord(c) < 0x20 for c in value):
        return DEFAULT_NEXT
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return DEFAULT_NEXT
    if parts.path in ("/login", "/logout"):
        return DEFAULT_NEXT
    return value


def _norm_host(host: str, scheme: str = "") -> str:
    host = host.strip().lower()
    if scheme == "https" and host.endswith(":443"):
        host = host[: -len(":443")]
    elif scheme == "http" and host.endswith(":80"):
        host = host[: -len(":80")]
    return host


def csrf_ok(request: Request) -> bool:
    """Fetch-Metadata first, then Origin vs Host/X-Forwarded-Host."""
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site == "same-origin"
    origin = request.headers.get("origin")
    if not origin or origin == "null":
        return False
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    origin_host = _norm_host(parts.netloc, parts.scheme)
    candidates = []
    for name in ("host", "x-forwarded-host"):
        value = request.headers.get(name)
        if value:
            # X-Forwarded-Host may be a comma list; the first hop is the client's.
            candidates.append(_norm_host(value.split(",")[0], parts.scheme))
    return origin_host in candidates


def is_authenticated(request: Request, policy: SecurityPolicy) -> bool:
    return auth.verify_session(
        request.cookies.get(auth.SESSION_COOKIE), policy.current_token()
    )


def _is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or proto == "https"


def _wants_html(request: Request) -> bool:
    return request.method in ("GET", "HEAD") and "text/html" in request.headers.get(
        "accept", ""
    )


def _request_target(request: Request) -> str:
    target = request.url.path
    if request.url.query:
        target += "?" + request.url.query
    return target


def _htmx_next(request: Request) -> str:
    current = request.headers.get("hx-current-url")
    if current:
        parts = urlsplit(current)
        if not parts.netloc or parts.netloc == request.url.netloc:
            return safe_next(parts.path + (f"?{parts.query}" if parts.query else ""))
    return DEFAULT_NEXT


def _private_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}


def render_login(
    request: Request, next_path: str, *, error: Optional[str] = None, status_code: int = 200
) -> HTMLResponse:
    response = request.app.state.templates.TemplateResponse(
        request,
        "login.html",
        {"next_path": safe_next(next_path), "error": error},
        status_code=status_code,
    )
    response.headers.update(_private_headers())
    return response


def _set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE,
        auth.session_value(token),
        max_age=auth.SESSION_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_is_https(request),
    )


def _login_success(request: Request, token: str, next_path: str) -> RedirectResponse:
    response = RedirectResponse(safe_next(next_path), status_code=303)
    response.headers.update(_private_headers())
    _set_session_cookie(response, request, token)
    return response


async def _security_middleware(request: Request, call_next):
    policy = _policy_for(request)
    if not policy.enabled:
        return await call_next(request)

    if request.method in UNSAFE_METHODS and not csrf_ok(request):
        _log.warning("csrf: rejected %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "cross-site request rejected"}, status_code=403)

    if is_exempt(request.url.path) or is_authenticated(request, policy):
        return await call_next(request)

    if request.headers.get("hx-request") == "true":
        login_url = "/login?next=" + quote(_htmx_next(request), safe="/")
        return Response(
            status_code=401, headers={"HX-Redirect": login_url, **_private_headers()}
        )
    if _wants_html(request):
        return render_login(request, _request_target(request), status_code=401)
    return JSONResponse(
        {"detail": "authentication required"}, status_code=401, headers=_private_headers()
    )


router = APIRouter()


@router.get("/login", include_in_schema=False)
async def login_page(
    request: Request,
    next_path: Optional[str] = Query(None, alias="next"),
    token: Optional[str] = None,
):
    policy = _policy_for(request)
    if not policy.enabled:
        return RedirectResponse(safe_next(next_path), status_code=303)
    if token is not None:
        # One-tap login link from `argos web token --url`: plant the cookie and
        # 303 away so the token disappears from the address bar and history.
        current = policy.current_token()
        if auth.verify_token(token, current):
            return _login_success(request, current, next_path or DEFAULT_NEXT)
        await asyncio.sleep(policy.failed_login_delay)
        return render_login(
            request, next_path or DEFAULT_NEXT, error="토큰이 맞지 않습니다.", status_code=401
        )
    if is_authenticated(request, policy):
        return RedirectResponse(safe_next(next_path), status_code=303)
    return render_login(request, next_path or DEFAULT_NEXT)


@router.post("/login", include_in_schema=False)
async def login_submit(request: Request):
    policy = _policy_for(request)
    form = await request.form()
    next_path = str(form.get("next") or DEFAULT_NEXT)
    if not policy.enabled:
        return RedirectResponse(safe_next(next_path), status_code=303)
    current = policy.current_token()
    if auth.verify_token(str(form.get("token") or ""), current):
        return _login_success(request, current, next_path)
    await asyncio.sleep(policy.failed_login_delay)
    return render_login(request, next_path, error="토큰이 맞지 않습니다.", status_code=401)


@router.post("/logout", include_in_schema=False)
async def logout(request: Request):
    response = RedirectResponse("/login", status_code=303)
    response.headers.update(_private_headers())
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return response


def install_security(app: FastAPI, policy: Optional[SecurityPolicy] = None) -> None:
    """Register the auth/CSRF middleware and /login, /logout routes.

    Call after every other ``add_middleware`` so this runs outermost.
    ``policy=None`` defers to :func:`default_policy` at request time
    (enabled, default token path).
    """
    app.state.security_policy = policy
    app.include_router(router)
    app.middleware("http")(_security_middleware)
