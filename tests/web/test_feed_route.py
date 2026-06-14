"""Route + template tests for the 관측 피드 screen (ARG-136).

These exercise the GET /, GET /feed and GET /feed/items handlers without a
live database: the per-request session dependency is overridden and
``fetch_feed`` is monkeypatched to return a canned ``FeedPage``. This keeps
the tests runnable on release.yml CI (no Postgres).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus
from argos.web.app import _get_session, build_web_app
from argos.web.services.feed import FeedItem, FeedPage


def _item(
    *,
    title: str,
    category: CategoryType | None = None,
    image_url: str | None = None,
    status: AssetStatus | None = None,
) -> FeedItem:
    return FeedItem(
        id=uuid.uuid4(),
        title=title,
        source_url="https://example.com/" + title.replace(" ", "-"),
        category=category,
        image_url=image_url,
        status=status,
        sort_at=datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc),
    )


def _client_with_feed(monkeypatch, page: FeedPage, capture: list | None = None) -> TestClient:
    """Build a TestClient whose feed route returns ``page`` without DB access."""
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_feed(session, *, category=None, cursor=None, limit=20):
        if capture is not None:
            capture.append({"category": category, "cursor": cursor, "limit": limit})
        return page

    monkeypatch.setattr("argos.web.app.fetch_feed", _fake_fetch_feed)
    return TestClient(app)


def test_root_redirects_to_feed(monkeypatch):
    client = _client_with_feed(monkeypatch, FeedPage(items=[], next_cursor=None))
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/feed"


def test_feed_renders_cards_with_titles(monkeypatch):
    page = FeedPage(
        items=[
            _item(title="Alpha Thing", category=CategoryType.ALPHA),
            _item(title="Mainstream Thing", category=CategoryType.MAINSTREAM),
        ],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    resp = client.get("/feed")
    assert resp.status_code == 200
    body = resp.text
    assert "Alpha Thing" in body
    assert "Mainstream Thing" in body
    # Full page renders the base layout (tabbar + doctype).
    assert "<!DOCTYPE html>" in body
    assert 'class="tabbar"' in body


def test_feed_renders_category_tags(monkeypatch):
    page = FeedPage(
        items=[
            _item(title="A", category=CategoryType.ALPHA),
            _item(title="M", category=CategoryType.MAINSTREAM),
        ],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "Alpha" in body
    assert "Mainstream" in body
    # Category-specific tag modifier classes from argos.css.
    assert "tag alpha" in body or "tag main" in body


def test_feed_shows_asset_status_badge(monkeypatch):
    page = FeedPage(
        items=[
            _item(title="Kept", category=CategoryType.MAINSTREAM, status=AssetStatus.KEEP),
            _item(title="Gone", category=CategoryType.ALPHA, status=AssetStatus.ARCHIVED),
        ],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "Keep" in body
    assert "Archived" in body


def test_feed_uses_og_image_when_present(monkeypatch):
    page = FeedPage(
        items=[_item(title="WithImg", image_url="https://img.example.com/x.png")],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "https://img.example.com/x.png" in body


def test_feed_falls_back_to_banner_without_image(monkeypatch):
    page = FeedPage(
        items=[_item(title="NoImg", category=CategoryType.ALPHA, image_url=None)],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    # No broken <img>/background pointing at a None/empty url.
    assert "url('None')" not in body
    assert "url()" not in body
    # A dedicated fallback cover class is rendered instead.
    assert "cover--fallback" in body


def test_feed_passes_category_filter_to_service(monkeypatch):
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    client.get("/feed?category=Alpha")
    assert capture and capture[-1]["category"] == "Alpha"


def test_feed_invalid_category_treated_as_all(monkeypatch):
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    resp = client.get("/feed?category=garbage")
    assert resp.status_code == 200
    assert capture[-1]["category"] is None


def test_feed_renders_load_more_when_next_cursor(monkeypatch):
    page = FeedPage(items=[_item(title="X")], next_cursor="CURSOR123")
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "load-more" in body
    assert "CURSOR123" in body
    assert "/feed/items" in body


def test_feed_no_load_more_when_no_next_cursor(monkeypatch):
    page = FeedPage(items=[_item(title="X")], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "load-more" not in body


def test_feed_items_fragment_is_partial_only(monkeypatch):
    page = FeedPage(items=[_item(title="FragItem")], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    resp = client.get("/feed/items")
    assert resp.status_code == 200
    body = resp.text
    assert "FragItem" in body
    # Fragment must NOT include the base layout.
    assert "<!DOCTYPE html>" not in body
    assert 'class="tabbar"' not in body


def test_feed_items_fragment_carries_category_in_load_more(monkeypatch):
    page = FeedPage(items=[_item(title="X")], next_cursor="NEXT9")
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed/items?category=Alpha").text
    assert "NEXT9" in body
    assert "category=Alpha" in body


def test_feed_items_fragment_passes_cursor_to_service(monkeypatch):
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    client.get("/feed/items?cursor=ABC")
    assert capture[-1]["cursor"] == "ABC"


def _client_real_feed() -> TestClient:
    """TestClient using the REAL fetch_feed with a None session.

    A malformed cursor fails in decode_cursor *before* any DB access, so the
    None session is never touched — letting us assert the route's cursor
    guard without Postgres. raise_server_exceptions=False so an unguarded
    ValueError surfaces as a 500 response rather than re-raising in the test.
    """
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session
    return TestClient(app, raise_server_exceptions=False)


def test_feed_malformed_cursor_returns_400_not_500():
    client = _client_real_feed()
    resp = client.get("/feed?cursor=not-a-valid-cursor")
    assert resp.status_code == 400


def test_feed_items_malformed_cursor_returns_400_not_500():
    client = _client_real_feed()
    resp = client.get("/feed/items?cursor=%%%bogus%%%")
    assert resp.status_code == 400
