"""Verify base.html structure and child-template inheritance (ARG-147).

Strategy: register a temporary child template + FastAPI route at
test time so we exercise the real Jinja2 environment from
build_web_app() instead of asserting against a string snapshot.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from argos.web.app import build_web_app


CHILD_TEMPLATE_NAME = "_arg147_test_child.html"
CHILD_TEMPLATE_BODY = """{% extends "base.html" %}
{% block content %}<main id="child-marker">CHILD CONTENT</main>{% endblock %}
"""


@pytest.fixture()
def app_with_child_template(tmp_path: Path):
    """Drop a test-only child template into the live templates dir.

    Cleaned up after the test so the working tree stays untouched.
    """
    templates_dir = Path(__file__).parent.parent.parent / "src" / "argos" / "web" / "templates"
    child = templates_dir / CHILD_TEMPLATE_NAME
    child.write_text(CHILD_TEMPLATE_BODY, encoding="utf-8")
    try:
        app = build_web_app()

        @app.get("/__test_child__", response_class=HTMLResponse)
        async def _render_child(request: Request) -> HTMLResponse:
            templates = app.state.templates
            return templates.TemplateResponse(
                request, CHILD_TEMPLATE_NAME, {}
            )

        yield TestClient(app)
    finally:
        child.unlink(missing_ok=True)


def test_base_renders_masthead(app_with_child_template: TestClient) -> None:
    html = app_with_child_template.get("/__test_child__").text
    assert "<header" in html and 'class="masthead"' in html
    assert "ARGOS" in html
    assert "/static/img/logo.svg" in html


def test_base_renders_bottom_tabbar_with_feed_and_portfolio(
    app_with_child_template: TestClient,
) -> None:
    html = app_with_child_template.get("/__test_child__").text
    assert 'class="tabbar"' in html
    assert "관측 피드" in html or "Feed" in html
    assert "포트폴리오" in html or "Portfolio" in html


def test_base_links_argos_css_and_htmx(app_with_child_template: TestClient) -> None:
    html = app_with_child_template.get("/__test_child__").text
    assert "/static/css/argos.css" in html
    assert "/static/js/htmx.min.js" in html


def test_base_emits_viewport_and_theme_color(app_with_child_template: TestClient) -> None:
    html = app_with_child_template.get("/__test_child__").text
    assert 'name="viewport"' in html
    assert 'name="theme-color"' in html
    assert "#0b0d12" in html


def test_child_block_content_is_rendered(app_with_child_template: TestClient) -> None:
    html = app_with_child_template.get("/__test_child__").text
    assert 'id="child-marker"' in html
    assert "CHILD CONTENT" in html


def test_base_includes_sky_and_grain_atmosphere(
    app_with_child_template: TestClient,
) -> None:
    html = app_with_child_template.get("/__test_child__").text
    assert 'class="sky"' in html
    assert 'class="grain"' in html
