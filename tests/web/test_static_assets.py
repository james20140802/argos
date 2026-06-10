"""Smoke tests that vendored static assets are present and served (ARG-145)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

STATIC_ROUTES = [
    "/static/img/logo.svg",
    "/static/fonts/Fraunces-Regular.woff2",
    "/static/fonts/Fraunces-SemiBold.woff2",
    "/static/fonts/GowunBatang-Regular.woff2",
    "/static/fonts/GowunBatang-Bold.woff2",
    "/static/fonts/IBMPlexSansKR-Regular.woff2",
    "/static/fonts/IBMPlexSansKR-SemiBold.woff2",
    "/static/fonts/IBMPlexMono-Regular.woff2",
    "/static/fonts/IBMPlexMono-Medium.woff2",
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


def test_woff2_files_are_nonempty(web_client: TestClient) -> None:
    for route in STATIC_ROUTES:
        if not route.endswith(".woff2"):
            continue
        response = web_client.get(route)
        assert len(response.content) > 1024, (
            f"{route} is suspiciously small ({len(response.content)} bytes); "
            "verify the font was actually downloaded, not stubbed."
        )
