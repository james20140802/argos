"""Filter/sort taps swap <main> in place instead of a full page load, and
responses are compressed (ARG-243 responsiveness pass). No DB required."""
from __future__ import annotations

import re
from pathlib import Path

from starlette.testclient import TestClient

from argos.web.app import _get_session, build_web_app
from argos.web.services.feed import FeedPage

PKG = Path(__file__).resolve().parents[2] / "src" / "argos" / "web"


def _feed_client(monkeypatch) -> TestClient:
    app = build_web_app()

    async def _session():
        yield None

    app.dependency_overrides[_get_session] = _session

    async def _page(session, **kw):
        return FeedPage(items=[], next_cursor=None)

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return []

    monkeypatch.setattr("argos.web.app.fetch_feed", _page)
    monkeypatch.setattr("argos.web.app.select_hero", _none)
    monkeypatch.setattr("argos.web.app.latest_feed_cursor", _none)
    monkeypatch.setattr("argos.web.app.fetch_activity", _empty)
    return TestClient(app)


def test_filter_and_sort_navs_are_boosted_into_main(monkeypatch):
    body = _feed_client(monkeypatch).get("/feed").text
    navs = re.findall(r"<nav class=\"(feed-sort[^\"]*|feed-filter)\"[^>]*>", body)
    assert len(navs) == 2
    for tag in re.findall(r"<nav class=\"(?:feed-sort|feed-filter)[^>]*>", body):
        assert "data-instant-nav" in tag
        assert 'hx-boost="true"' in tag
        assert 'hx-select="#main"' in tag and 'hx-target="#main"' in tag
    assert '<main id="main"' in body
    # Still real links: without JS they navigate normally.
    assert 'href="/feed?category=Mainstream"' in body


def test_html_is_gzipped(monkeypatch):
    resp = _feed_client(monkeypatch).get("/feed", headers={"Accept-Encoding": "gzip"})
    assert resp.headers.get("content-encoding") == "gzip"


def test_poll_pill_is_looked_up_live_not_cached():
    js = (PKG / "static" / "js" / "feed-poll.js").read_text(encoding="utf-8")
    assert "var pill = document.querySelector" not in js
    assert 'closest("[data-new-items-pill]")' in js


def test_instant_nav_is_loaded_and_precached():
    sw = (PKG / "assets" / "sw.js").read_text(encoding="utf-8")
    assert "/static/js/instant-nav.js" in sw
    assert "/static/js/view-transitions.js" in sw


def test_css_and_js_urls_carry_a_content_version(monkeypatch):
    """A changed stylesheet must be a new URL, or a cached old copy keeps
    rendering (the iPad showed a fixed layout bug for this reason)."""
    body = _feed_client(monkeypatch).get("/feed").text
    m = re.search(r'href="/static/css/argos\.css\?v=([0-9a-f]{10})"', body)
    assert m, "stylesheet URL has no content version"
    assert f'/static/js/instant-nav.js?v={m.group(1)}"' in body


def test_static_files_are_revalidated(monkeypatch):
    resp = _feed_client(monkeypatch).get("/static/css/argos.css")
    assert resp.headers.get("cache-control") == "no-cache"


def test_sw_precache_bypasses_http_cache():
    sw = (PKG / "assets" / "sw.js").read_text(encoding="utf-8")
    assert "cache: 'reload'" in sw
