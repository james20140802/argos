"""Tests for the PWA install layer (ARG-140).

Covers:
* /manifest.webmanifest is served with correct MIME type and contents
* /sw.js is served at origin root with Service-Worker-Allowed: / and JS MIME
* PWA icon PNGs (192, 512, maskable 512) are present and non-trivial
* base.html links the manifest, A2HS meta tags, apple-touch-icon, and the
  registration script
* The SW registration script guards on isSecureContext so HTTP loads do not
  attempt to register a service worker
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import argos.web


PACKAGE_DIR = Path(argos.web.__file__).parent
ICON_DIR = PACKAGE_DIR / "static" / "img" / "icons"
SW_REGISTER_PATH = PACKAGE_DIR / "static" / "js" / "sw-register.js"


# ---------------------------------------------------------------------------
# /manifest.webmanifest
# ---------------------------------------------------------------------------


def test_manifest_route_returns_200(web_client: TestClient) -> None:
    resp = web_client.get("/manifest.webmanifest")
    assert resp.status_code == 200


def test_manifest_content_type_is_webmanifest(web_client: TestClient) -> None:
    resp = web_client.get("/manifest.webmanifest")
    assert resp.headers["content-type"].startswith("application/manifest+json")


def test_manifest_required_fields(web_client: TestClient) -> None:
    data = web_client.get("/manifest.webmanifest").json()
    assert data["name"]
    assert data["short_name"]
    assert data["start_url"] == "/feed"
    assert data["display"] == "standalone"
    assert data["theme_color"].lower() == "#0b0d12"
    assert data["background_color"].lower() == "#0b0d12"


def test_manifest_lists_icons_with_192_512_and_maskable(web_client: TestClient) -> None:
    data = web_client.get("/manifest.webmanifest").json()
    icons = data["icons"]
    sizes = {icon["sizes"] for icon in icons}
    assert "192x192" in sizes
    assert "512x512" in sizes
    purposes = {icon.get("purpose", "any") for icon in icons}
    assert any("maskable" in p for p in purposes)
    for icon in icons:
        assert icon["src"].startswith("/static/img/icons/")
        assert icon["type"] == "image/png"


# ---------------------------------------------------------------------------
# /sw.js (service worker at root scope)
# ---------------------------------------------------------------------------


def test_sw_route_returns_200(web_client: TestClient) -> None:
    resp = web_client.get("/sw.js")
    assert resp.status_code == 200


def test_sw_route_serves_javascript_mime(web_client: TestClient) -> None:
    resp = web_client.get("/sw.js")
    assert "javascript" in resp.headers["content-type"].lower()


def test_sw_route_sets_service_worker_allowed_root_scope(web_client: TestClient) -> None:
    """Without Service-Worker-Allowed: /, a SW served from /sw.js would still
    be scoped to /, but explicit declaration documents the intent and is
    required when serving from a non-root path. We set it to / for clarity
    and forward-compat with any future relocation."""
    resp = web_client.get("/sw.js")
    assert resp.headers.get("service-worker-allowed") == "/"


def test_sw_body_contains_install_and_fetch_handlers(web_client: TestClient) -> None:
    body = web_client.get("/sw.js").text
    assert "addEventListener('install'" in body or 'addEventListener("install"' in body
    assert "addEventListener('fetch'" in body or 'addEventListener("fetch"' in body


def test_sw_body_caches_app_shell(web_client: TestClient) -> None:
    body = web_client.get("/sw.js").text
    # App-shell should include the feed entry point + core static assets.
    assert "/feed" in body
    assert "/static/css/argos.css" in body


def test_sw_navigation_caching_gated_to_app_shell_routes(
    web_client: TestClient,
) -> None:
    """Navigation SWR must be restricted to the /feed + /portfolio shell so
    state-sensitive detail pages (/item/{id}) are never cached and served stale.
    """
    body = web_client.get("/sw.js").text
    assert "APP_SHELL_ROUTES" in body
    # The navigate branch must be gated by the route allowlist, not fire for
    # every navigation request.
    assert "req.mode === 'navigate' && APP_SHELL_ROUTES.includes(url.pathname)" in body


# ---------------------------------------------------------------------------
# Icon PNGs on disk + served via /static/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["icon-192.png", "icon-512.png", "icon-maskable-512.png"],
)
def test_icon_file_exists_and_is_nontrivial(filename: str) -> None:
    p = ICON_DIR / filename
    assert p.exists(), f"missing icon {p}"
    assert p.stat().st_size > 512, f"icon {p} suspiciously small"


@pytest.mark.parametrize(
    "route",
    [
        "/static/img/icons/icon-192.png",
        "/static/img/icons/icon-512.png",
        "/static/img/icons/icon-maskable-512.png",
    ],
)
def test_icon_route_returns_200(web_client: TestClient, route: str) -> None:
    resp = web_client.get(route)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # PNG magic number
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# base.html wiring: manifest link, A2HS meta tags, registration script
# ---------------------------------------------------------------------------


def test_base_template_links_manifest() -> None:
    html = (PACKAGE_DIR / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'rel="manifest"' in html
    assert "/manifest.webmanifest" in html


def test_base_template_has_apple_touch_icon() -> None:
    html = (PACKAGE_DIR / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'rel="apple-touch-icon"' in html
    assert "/static/img/icons/" in html


def test_base_template_has_a2hs_meta_tags() -> None:
    html = (PACKAGE_DIR / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'name="apple-mobile-web-app-capable"' in html
    assert 'name="apple-mobile-web-app-status-bar-style"' in html
    assert 'name="apple-mobile-web-app-title"' in html


def test_base_template_includes_sw_register_script() -> None:
    html = (PACKAGE_DIR / "templates" / "base.html").read_text(encoding="utf-8")
    assert "/static/js/sw-register.js" in html


# ---------------------------------------------------------------------------
# sw-register.js: HTTPS-only guard so /feed over HTTP still renders cleanly
# ---------------------------------------------------------------------------


def test_sw_register_file_exists_and_is_served(web_client: TestClient) -> None:
    assert SW_REGISTER_PATH.exists()
    resp = web_client.get("/static/js/sw-register.js")
    assert resp.status_code == 200


def test_sw_register_guards_on_secure_context() -> None:
    body = SW_REGISTER_PATH.read_text(encoding="utf-8")
    # Must short-circuit when not a secure context (HTTP tailnet IP path).
    assert "isSecureContext" in body
    # Must check for serviceWorker support.
    assert "serviceWorker" in body
    # Must point at the root-scope SW.
    assert "/sw.js" in body


def test_sw_register_does_not_throw_on_missing_service_worker() -> None:
    """The script must defensively skip when navigator.serviceWorker is undefined
    (older browsers / locked-down WebViews) instead of throwing."""
    body = SW_REGISTER_PATH.read_text(encoding="utf-8")
    # Either a guard reference or a typeof check is acceptable.
    assert ("'serviceWorker' in navigator" in body
            or '"serviceWorker" in navigator' in body
            or "typeof navigator" in body)


# ---------------------------------------------------------------------------
# Cross-check: manifest's listed icons all resolve under /static/
# ---------------------------------------------------------------------------


def test_every_manifest_icon_is_actually_served(web_client: TestClient) -> None:
    data = web_client.get("/manifest.webmanifest").json()
    for icon in data["icons"]:
        r = web_client.get(icon["src"])
        assert r.status_code == 200, f"manifest icon {icon['src']} not served"
        assert r.headers["content-type"] == "image/png"


def test_manifest_parses_as_json(web_client: TestClient) -> None:
    """A malformed manifest silently disables installability — guard with
    a strict parse on the raw bytes."""
    raw = web_client.get("/manifest.webmanifest").content
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
