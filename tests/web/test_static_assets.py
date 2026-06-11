"""Smoke tests that vendored static assets are present and served (ARG-145)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Routes that must respond 200 once T1 lands. Keep in sync with the files
# vendored under src/argos/web/static/.
STATIC_ROUTES = [
    "/static/css/argos.css",
    "/static/img/logo.svg",
    "/static/fonts/Fraunces-Regular.woff2",
    "/static/fonts/Fraunces-SemiBold.woff2",
    "/static/fonts/GowunBatang-Regular.woff2",
    "/static/fonts/GowunBatang-Bold.woff2",
    "/static/fonts/IBMPlexSansKR-Regular.woff2",
    "/static/fonts/IBMPlexSansKR-SemiBold.woff2",
    "/static/fonts/IBMPlexMono-Regular.woff2",
    "/static/fonts/IBMPlexMono-Medium.woff2",
    "/static/js/htmx.min.js",
]

FONT_ROUTES = [r for r in STATIC_ROUTES if r.endswith(".woff2")]


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


@pytest.mark.parametrize("route", FONT_ROUTES)
def test_woff2_file_is_nonempty(web_client: TestClient, route: str) -> None:
    """Guard against zero-byte placeholders sneaking into the vendored set."""
    response = web_client.get(route)
    assert len(response.content) > 1024, (
        f"{route} is suspiciously small ({len(response.content)} bytes); "
        "verify the font was actually downloaded, not stubbed."
    )


def test_argos_css_contains_required_tokens(web_client: TestClient) -> None:
    """argos.css must declare the Midnight Observatory token set."""
    body = web_client.get("/static/css/argos.css").text
    for token in ("--bg", "--ink", "--brass", "--main", "--alpha"):
        assert token in body, f"missing token {token}"
    assert "#C9A86A" in body
    assert "@font-face" in body
    assert "Fraunces" in body
    assert "Gowun Batang" in body
    assert "IBM Plex Sans KR" in body
    assert "IBM Plex Mono" in body
    assert ".card" in body or ".ncard" in body
    assert "conic-gradient" in body
    assert "backdrop-filter" in body
    assert "prefers-reduced-motion" in body
