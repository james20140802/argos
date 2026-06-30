"""Favicon-only render tests (ARG-178 / T3).

Asserts the three-way image branch in _feed_card.html and item_detail.html:
  1. favicon-only (image_url ends with /favicon.ico) → gradient + favicon chip + domain
  2. real-image → normal cover__img / detail-hero__img
  3. no image_url → ◎ fallback glyph (cover__glyph / detail-hero__glyph class)

NOTE: base.html tabbar independently uses ◎ as a navigation icon, so tests
check for branch-specific CSS classes (cover__glyph, detail-hero__glyph) rather
than raw glyph character equality to stay robust against layout chrome.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus
from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import ItemDetailView
from argos.web.services.feed import FeedItem, FeedPage


# ──────────────────────────────────────────────────────
# Feed-card helpers (mirrors test_feed_route.py)
# ──────────────────────────────────────────────────────

def _feed_item(
    *,
    title: str = "Test Item",
    image_url: str | None = None,
    source_url: str = "https://example.com/post",
    category: CategoryType | None = CategoryType.ALPHA,
    status: AssetStatus | None = None,
    summary: str | None = None,
) -> FeedItem:
    return FeedItem(
        id=uuid.uuid4(),
        title=title,
        source_url=source_url,
        category=category,
        image_url=image_url,
        summary=summary,
        status=status,
        sort_at=datetime(2026, 6, 30, 3, 0, tzinfo=timezone.utc),
    )


def _client_with_feed(monkeypatch, page: FeedPage) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_feed(session, *, category=None, cursor=None, limit=20):
        return page

    monkeypatch.setattr("argos.web.app.fetch_feed", _fake_fetch_feed)
    return TestClient(app)


# ──────────────────────────────────────────────────────
# Detail-route helpers (mirrors test_item_detail_route.py)
# ──────────────────────────────────────────────────────

def _detail_view(
    *,
    image_url: str | None = None,
    source_url: str = "https://example.com/post",
    category: CategoryType | None = CategoryType.ALPHA,
) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title="Test Title",
        source_url=source_url,
        image_url=image_url,
        summary="Summary here.",
        category=category,
        trust_score=0.75,
        published_at=None,
    )


def _client_with_detail(monkeypatch, view: ItemDetailView) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_item_detail(session, item_id):
        return view

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch_item_detail)
    return TestClient(app)


# ──────────────────────────────────────────────────────
# Feed-card tests
# ──────────────────────────────────────────────────────

def test_feed_favicon_only_renders_chip_and_domain(monkeypatch):
    """favicon-only item: gradient + favicon chip + domain text."""
    item = _feed_item(
        image_url="https://example.com/favicon.ico",
        source_url="https://example.com/post",
    )
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    response = client.get("/feed")
    assert response.status_code == 200
    body = response.text

    assert "cover--favicon" in body
    assert "favicon-chip" in body
    # Domain text must render inside the favicon chip. Use a regex anchored on
    # the chip element rather than a bare ``"example.com" in body`` substring
    # check — the latter trips CodeQL's incomplete-URL-sanitization heuristic
    # even though this is rendered HTML, not a security gate.
    assert re.search(r'favicon-chip__domain"[^>]*>\s*example\.com', body)
    # The cover__glyph class only appears in the fallback branch — not here.
    assert "cover__glyph" not in body


def test_feed_favicon_only_does_not_render_cover_img(monkeypatch):
    """favicon-only item must NOT use the regular cover__img path."""
    item = _feed_item(
        image_url="https://example.com/favicon.ico",
        source_url="https://example.com/post",
    )
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    response = client.get("/feed")
    assert response.status_code == 200
    body = response.text

    assert "cover--favicon" in body
    # cover__img is the class used for a real-image cover; must not appear
    assert "cover__img" not in body


def test_feed_real_image_renders_cover_img(monkeypatch):
    """Real-image item: normal cover__img, no favicon branch."""
    item = _feed_item(
        image_url="https://cdn.example.com/hero.jpg",
        source_url="https://example.com/post",
    )
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    response = client.get("/feed")
    assert response.status_code == 200
    body = response.text

    assert "cover__img" in body
    assert "cover--favicon" not in body
    assert "cover__glyph" not in body


def test_feed_no_image_renders_fallback_glyph(monkeypatch):
    """No image_url → cover__glyph fallback class; no favicon branch."""
    item = _feed_item(image_url=None)
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    response = client.get("/feed")
    assert response.status_code == 200
    body = response.text

    assert "cover__glyph" in body
    assert "cover--fallback" in body
    assert "cover--favicon" not in body


# ──────────────────────────────────────────────────────
# Detail-hero tests
# ──────────────────────────────────────────────────────

def test_detail_favicon_only_renders_chip_and_domain(monkeypatch):
    """favicon-only detail page: gradient + favicon chip + domain text."""
    view = _detail_view(
        image_url="https://example.com/favicon.ico",
        source_url="https://example.com/post",
    )
    client = _client_with_detail(monkeypatch, view)
    response = client.get(f"/item/{view.id}")
    assert response.status_code == 200
    body = response.text

    assert "detail-hero--favicon" in body
    assert "favicon-chip" in body
    # Domain text must render inside the favicon chip. Use a regex anchored on
    # the chip element rather than a bare ``"example.com" in body`` substring
    # check — the latter trips CodeQL's incomplete-URL-sanitization heuristic
    # even though this is rendered HTML, not a security gate.
    assert re.search(r'favicon-chip__domain"[^>]*>\s*example\.com', body)
    # detail-hero__glyph class only appears in the fallback branch.
    assert "detail-hero__glyph" not in body


def test_detail_real_image_renders_hero_img(monkeypatch):
    """Real-image detail page: normal detail-hero__img, no favicon branch."""
    view = _detail_view(
        image_url="https://cdn.example.com/hero.jpg",
        source_url="https://example.com/post",
    )
    client = _client_with_detail(monkeypatch, view)
    response = client.get(f"/item/{view.id}")
    assert response.status_code == 200
    body = response.text

    assert "detail-hero__img" in body
    assert "detail-hero--favicon" not in body
    assert "detail-hero__glyph" not in body


def test_detail_no_image_renders_fallback_glyph(monkeypatch):
    """No image_url on detail page → detail-hero__glyph fallback; no favicon branch."""
    view = _detail_view(image_url=None)
    client = _client_with_detail(monkeypatch, view)
    response = client.get(f"/item/{view.id}")
    assert response.status_code == 200
    body = response.text

    assert "detail-hero__glyph" in body
    assert "detail-hero--fallback" in body
    assert "detail-hero--favicon" not in body


# ──────────────────────────────────────────────────────
# Domain-filter: security invariant
# ──────────────────────────────────────────────────────

def test_feed_favicon_url_escaped_in_img_src_not_css(monkeypatch):
    """favicon.ico URL must only appear in an <img src>, not inline CSS."""
    item = _feed_item(
        image_url="https://example.com/favicon.ico",
        source_url="https://example.com/post",
    )
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    response = client.get("/feed")
    assert response.status_code == 200
    body = response.text

    assert "background-image: url(" not in body
    assert "url('https://example.com/favicon.ico')" not in body
    assert '<img' in body
