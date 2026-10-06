"""Smoke tests that vendored static assets are present and served (ARG-145)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Routes that must respond 200. Keep in sync with the files vendored under
# src/argos/web/static/. ARG-243 dropped the vendored web fonts in favour of
# the platform system font (SF on Apple devices), so no .woff2 remain.
STATIC_ROUTES = [
    "/static/css/argos.css",
    "/static/img/logo.svg",
    "/static/js/htmx.min.js",
]


@pytest.mark.parametrize("route", STATIC_ROUTES)
def test_static_asset_route_returns_200(web_client: TestClient, route: str) -> None:
    response = web_client.get(route)
    assert response.status_code == 200, (
        f"{route} returned {response.status_code}; vendored asset missing?"
    )


def test_logo_svg_is_radar_mark(web_client: TestClient) -> None:
    """Mark B (radar) from docs/design/argos-web-pwa-logo-marks.html."""
    body = web_client.get("/static/img/logo.svg").text
    assert "<svg" in body
    assert 'viewBox="0 0 40 40"' in body
    assert 'r="15"' in body
    assert 'r="9"' in body
    assert 'r="2.2"' in body
    assert 'fill="#C9A86A"' in body


def test_argos_css_contains_required_tokens(web_client: TestClient) -> None:
    """argos.css must declare the design token set (ARG-243 Apple redesign)."""
    body = web_client.get("/static/css/argos.css").text
    for token in ("--bg", "--ink", "--accent", "--main", "--alpha", "--spring"):
        assert token in body, f"missing token {token}"
    # System font stack — no vendored @font-face any more.
    assert "-apple-system" in body
    assert "@font-face" not in body
    assert "prefers-color-scheme: dark" in body
    assert "@view-transition" in body
    assert ".card" in body or ".ncard" in body
    assert "conic-gradient" in body
    assert "backdrop-filter" in body
    assert "prefers-reduced-motion" in body
