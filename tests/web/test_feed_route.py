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
    summary: str | None = None,
    trust_score: float | None = None,
) -> FeedItem:
    return FeedItem(
        id=uuid.uuid4(),
        title=title,
        source_url="https://example.com/" + title.replace(" ", "-"),
        category=category,
        image_url=image_url,
        summary=summary,
        status=status,
        trust_score=trust_score,
        sort_at=datetime(2026, 6, 14, 3, 0, tzinfo=timezone.utc),
    )


def _client_with_feed(
    monkeypatch,
    page: FeedPage,
    capture: list | None = None,
    *,
    hero_id=None,
) -> TestClient:
    """Build a TestClient whose feed route returns ``page`` without DB access.

    ``hero_id`` stands in for ``select_hero`` (ARG-213) — the real function
    needs a live session, so every first-page render must have it faked here
    (defaulting to no hero) or the None test session would blow up.
    """
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_feed(
        session, *, category=None, cursor=None, limit=20, sort="recommended"
    ):
        if capture is not None:
            capture.append(
                {"category": category, "cursor": cursor, "limit": limit, "sort": sort}
            )
        return page

    monkeypatch.setattr("argos.web.app.fetch_feed", _fake_fetch_feed)

    async def _fake_fetch_activity(session, limit=12):
        return []

    monkeypatch.setattr("argos.web.app.fetch_activity", _fake_fetch_activity)

    async def _fake_select_hero(session, *, category=None):
        return hero_id

    monkeypatch.setattr("argos.web.app.select_hero", _fake_select_hero)
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


def test_feed_cards_link_to_in_app_reader(monkeypatch):
    """Card taps must open the in-app reader (ARG-138), not bounce to the
    external source_url — that bounce is exactly the loss the detail page
    exists to prevent."""
    item = _item(title="Linkable", category=CategoryType.ALPHA)
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert f'href="/item/{item.id}"' in body
    # Cards no longer point straight at the external source.
    assert f'href="{item.source_url}"' not in body


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
    # Category now renders as a tinted editorial eyebrow (observation-log
    # redesign) rather than a pill tag.
    assert "eyebrow--alpha" in body or "eyebrow--mainstream" in body


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


def test_feed_shows_trust_score_dial(monkeypatch):
    item = _item(title="Trusted", category=CategoryType.ALPHA, trust_score=0.87)
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert 'class="trust-dial trust-dial--sm' in body
    # Ring-only at feed scale (ARG-189): the value rides the conic sweep and the
    # accessible label/tooltip, not a cramped inner number ("100" overlapped the
    # ring). So no __face is rendered here, but the % stays reachable.
    assert "trust-dial__face" not in body
    assert "신뢰도 87%" in body
    assert "--p: 87" in body


def test_feed_omits_trust_score_when_none(monkeypatch):
    item = _item(title="Untrusted", category=CategoryType.ALPHA, trust_score=None)
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "trust-dial" not in body


def test_feed_renders_keep_pass_controls(monkeypatch):
    """The Keep/Pass action endpoints must be reachable from the initial feed
    render (ARG-139) — not only after an HTMX swap. Regression guard for the
    controls living in an unincluded partial."""
    item = _item(title="Actionable", category=CategoryType.ALPHA)
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    # A lone first-page item is the featured hero, so its action URLs carry the
    # ?featured=1 suffix — match the path prefix to stay agnostic to it.
    assert f'hx-post="/items/{item.id}/keep' in body
    assert f'hx-post="/items/{item.id}/pass' in body
    assert f'id="feed-card-{item.id}"' in body


def test_feed_items_fragment_renders_keep_pass_controls(monkeypatch):
    """The bare /feed/items fragment (HTMX cursor pagination) must also carry
    the action controls so paginated-in cards stay actionable."""
    item = _item(title="FragActionable", category=CategoryType.ALPHA)
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed/items").text
    assert f'hx-post="/items/{item.id}/keep"' in body
    assert f'hx-post="/items/{item.id}/pass"' in body


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


def test_feed_image_url_rendered_safely_not_in_inline_css(monkeypatch):
    # A crawled og:image URL is attacker-controllable. A single quote must
    # not break out of a quoted url() in a style attribute.
    payload = "https://evil.example/a'); } body { display:none } /*"
    page = FeedPage(items=[_item(title="Pwn", image_url=payload)], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    # No untrusted URL injected into inline CSS at all.
    assert "background-image: url(" not in body
    # The raw CSS-breakout sequence must not appear unescaped.
    assert "'); } body" not in body
    # Rendered instead as an (HTML-escaped) <img src>.
    assert "<img" in body


# --------------------------------------------------------------------- #
# ARG-175 (T2) — magazine grid: summary line + featured hero
# --------------------------------------------------------------------- #

def test_feed_renders_summary_line_when_present(monkeypatch):
    page = FeedPage(
        items=[_item(title="Has Summary", summary="이것은 한 줄 요약입니다.")],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "이것은 한 줄 요약입니다." in body
    assert 'class="summary"' in body


def test_feed_omits_summary_when_absent(monkeypatch):
    page = FeedPage(items=[_item(title="No Summary", summary=None)], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    # No empty summary element is rendered when the column is null.
    assert 'class="summary"' not in body


def test_feed_summary_is_html_escaped(monkeypatch):
    # summary is triage-generated, but defense in depth: never inject raw HTML.
    page = FeedPage(
        items=[_item(title="Pwn", summary="<script>alert(1)</script>")],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_feed_first_card_is_featured_on_first_page(monkeypatch):
    first = _item(title="Hero Item", category=CategoryType.ALPHA)
    second = _item(title="Plain Item", category=CategoryType.MAINSTREAM)
    page = FeedPage(items=[first, second], next_cursor=None)
    client = _client_with_feed(monkeypatch, page, hero_id=first.id)
    body = client.get("/feed").text
    # Exactly one featured hero, and it is the item select_hero named.
    assert body.count("card--featured") == 1
    assert f'card--featured" id="feed-card-{first.id}"' in body
    assert f'card--featured" id="feed-card-{second.id}"' not in body


def test_feed_load_more_fragment_has_no_featured_hero(monkeypatch):
    """The HTMX 더 보기 fragment is reused for every subsequent page, so its
    index-0 item must NOT become a second hero mid-scroll (AC)."""
    page = FeedPage(
        items=[_item(title="FragFirst"), _item(title="FragSecond")],
        next_cursor=None,
    )
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed/items").text
    assert "card--featured" not in body


def test_feed_with_cursor_has_no_featured_hero(monkeypatch):
    """A direct hit on /feed?cursor=<token> is a mid-feed page (browser
    history / shared link), so its index-0 item must NOT be promoted to the
    hero slot — only the genuine first page (no cursor) gets a hero."""
    page = FeedPage(items=[_item(title="MidFeed")], next_cursor=None)
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed?cursor=SOMETOKEN").text
    assert "card--featured" not in body


def test_featured_card_action_buttons_carry_featured_flag(monkeypatch):
    """The featured hero's Keep/Pass buttons must post ?featured=1 so the
    swapped-in card re-renders as the hero (not a collapsed grid cell);
    standard cards must not carry the flag."""
    first = _item(title="HeroItem")
    second = _item(title="PlainItem")
    page = FeedPage(items=[first, second], next_cursor=None)
    client = _client_with_feed(monkeypatch, page, hero_id=first.id)
    body = client.get("/feed").text
    assert f'/items/{first.id}/keep?featured=1' in body
    assert f'/items/{first.id}/pass?featured=1' in body
    # The non-featured card keeps the plain action URLs.
    assert f'/items/{second.id}/keep"' in body
    assert f'/items/{second.id}/keep?featured=1' not in body


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


# --------------------------------------------------------------------- #
# ARG-213 — recommended-default sort, ?sort=latest toggle, id-based hero
# --------------------------------------------------------------------- #


def test_feed_default_sort_is_recommended(monkeypatch):
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    client.get("/feed")
    assert capture[-1]["sort"] == "recommended"


def test_feed_sort_latest_passed_to_service(monkeypatch):
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    client.get("/feed?sort=latest")
    assert capture[-1]["sort"] == "latest"


def test_feed_invalid_sort_falls_back_to_recommended(monkeypatch):
    """AC is deliberately lenient here: only a cross-sort *cursor* is a 400 —
    a bogus ``?sort=`` value just falls back to recommended."""
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    resp = client.get("/feed?sort=garbage")
    assert resp.status_code == 200
    assert capture[-1]["sort"] == "recommended"


def test_feed_items_default_sort_is_recommended(monkeypatch):
    capture: list = []
    client = _client_with_feed(
        monkeypatch, FeedPage(items=[], next_cursor=None), capture=capture
    )
    client.get("/feed/items")
    assert capture[-1]["sort"] == "recommended"


def test_feed_sort_toggle_renders_both_links(monkeypatch):
    client = _client_with_feed(monkeypatch, FeedPage(items=[], next_cursor=None))
    body = client.get("/feed").text
    assert "추천순" in body
    assert "최신순" in body
    assert 'href="/feed?sort=latest"' in body


def test_feed_sort_toggle_marks_recommended_active_by_default(monkeypatch):
    client = _client_with_feed(monkeypatch, FeedPage(items=[], next_cursor=None))
    body = client.get("/feed").text
    assert "추천순</a>" in body
    # The recommended link is the active one (no ?sort= means recommended).
    idx = body.index("추천순</a>")
    chip_start = body.rfind("<a class=", 0, idx)
    assert "is-active" in body[chip_start:idx]


def test_feed_sort_toggle_marks_latest_active(monkeypatch):
    client = _client_with_feed(monkeypatch, FeedPage(items=[], next_cursor=None))
    body = client.get("/feed?sort=latest").text
    idx = body.index("최신순</a>")
    chip_start = body.rfind("<a class=", 0, idx)
    assert "is-active" in body[chip_start:idx]


def test_feed_items_fragment_carries_sort_in_load_more(monkeypatch):
    page = FeedPage(items=[_item(title="X")], next_cursor="NEXT9")
    client = _client_with_feed(monkeypatch, page)
    body = client.get("/feed/items?sort=latest").text
    assert "sort=latest" in body


def test_feed_hero_is_selected_by_id_not_position(monkeypatch):
    """Hero must be the item whose id matches ``select_hero``'s result — NOT
    positional index-0 — proving position-independence (ARG-213)."""
    first = _item(title="Not The Hero")
    second = _item(title="The Real Hero")
    page = FeedPage(items=[first, second], next_cursor=None)
    client = _client_with_feed(monkeypatch, page, hero_id=second.id)
    body = client.get("/feed").text
    assert body.count("card--featured") == 1
    assert f'card--featured" id="feed-card-{second.id}"' in body
    assert f'card--featured" id="feed-card-{first.id}"' not in body


def test_feed_no_hero_when_select_hero_returns_none(monkeypatch):
    page = FeedPage(items=[_item(title="Solo")], next_cursor=None)
    client = _client_with_feed(monkeypatch, page, hero_id=None)
    body = client.get("/feed").text
    assert "card--featured" not in body


def test_feed_items_fragment_never_has_hero_even_with_hero_id(monkeypatch):
    """The /feed/items 더보기 fragment always renders with first_page=False,
    so it must never show a hero even if a hero_id happens to be supplied."""
    item = _item(title="FragItem")
    page = FeedPage(items=[item], next_cursor=None)
    client = _client_with_feed(monkeypatch, page, hero_id=item.id)
    body = client.get("/feed/items").text
    assert "card--featured" not in body
