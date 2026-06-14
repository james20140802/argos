"""Verify base.html structure and child-template inheritance (ARG-147).

Strategy: register a temporary child template + FastAPI route at
test time so we exercise the real Jinja2 environment from
build_web_app() instead of asserting against a string snapshot.

The child template lives in ``tmp_path`` (never the source tree). The
app's Jinja2 loader is pointed at ``[tmp_path, <package templates>]`` so
``base.html`` still resolves from the package while the child resolves
from the temp dir. Because nothing is written under ``src/``, a hard
kill mid-test cannot leave a stray template in the committed tree.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

import argos.web
from argos.web.app import build_web_app


CHILD_TEMPLATE_NAME = "_arg147_test_child.html"
CHILD_TEMPLATE_BODY = """{% extends "base.html" %}
{% block content %}<main id="child-marker">CHILD CONTENT</main>{% endblock %}
"""


@pytest.fixture()
def app_with_child_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build the web app with a tmp-only child template mounted alongside base.

    Exposes the child at ``/__test_child__`` plus the real tab paths
    ``/feed`` and ``/portfolio`` so base.html's ``aria-current`` logic
    (which keys off ``request.url.path``) is actually exercised.

    Both ``/feed`` (ARG-136) and ``/portfolio`` (ARG-137) are real routes, so
    rather than stub them we make them render offline (empty pages, no DB) —
    they render their respective templates which both extend ``base.html``.
    """
    child = tmp_path / CHILD_TEMPLATE_NAME
    child.write_text(CHILD_TEMPLATE_BODY, encoding="utf-8")

    package_templates = Path(argos.web.__file__).parent / "templates"
    app = build_web_app()
    # Override the package-only loader with [tmp_path, package] so the
    # child resolves from tmp_path and base.html from the package dir.
    app.state.templates = Jinja2Templates(directory=[tmp_path, package_templates])

    async def _render_child(request: Request) -> HTMLResponse:
        return app.state.templates.TemplateResponse(request, CHILD_TEMPLATE_NAME, {})

    # Make the real /feed and /portfolio routes render without Postgres.
    from argos.web.app import _get_session
    from argos.web.services.feed import FeedPage
    from argos.web.services.portfolio import PortfolioView

    async def _fake_session():
        yield None

    async def _empty_feed(session, *, category=None, cursor=None, limit=20):
        return FeedPage(items=[], next_cursor=None)

    async def _empty_portfolio(session, *, category=None, sort="recency"):
        return PortfolioView(active=[], quiet=[], category=None, sort="recency")

    app.dependency_overrides[_get_session] = _fake_session
    monkeypatch.setattr("argos.web.app.fetch_feed", _empty_feed)
    monkeypatch.setattr("argos.web.app.fetch_portfolio", _empty_portfolio)

    app.get("/__test_child__", response_class=HTMLResponse)(_render_child)

    return TestClient(app)


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


@pytest.mark.parametrize(
    ("path", "active_href", "inactive_href"),
    [
        ("/feed", "/feed", "/portfolio"),
        ("/portfolio", "/portfolio", "/feed"),
    ],
)
def test_active_tab_marked_aria_current(
    app_with_child_template: TestClient,
    path: str,
    active_href: str,
    inactive_href: str,
) -> None:
    """The tab matching the current path carries aria-current; the other doesn't."""
    html = app_with_child_template.get(path).text
    # Exactly one tab is marked current.
    assert html.count('aria-current="page"') == 1
    # The active tab's anchor carries it; the inactive one does not.
    assert re.search(rf'href="{re.escape(active_href)}"[^>]*aria-current="page"', html)
    assert not re.search(
        rf'href="{re.escape(inactive_href)}"[^>]*aria-current="page"', html
    )
