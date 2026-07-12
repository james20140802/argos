"""Route + template tests for the 포트폴리오 screen (ARG-154).

These exercise the GET /portfolio handler without a live database: the
per-request session dependency is overridden and ``fetch_portfolio`` is
monkeypatched to return a canned ``PortfolioView``. This keeps the tests
runnable on release.yml CI (no Postgres).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.web.app import _get_session, build_web_app
from argos.web.services.portfolio import PortfolioAsset, PortfolioView


def _asset(
    *,
    title: str,
    category: CategoryType | None = CategoryType.MAINSTREAM,
    image_url: str | None = None,
    signal_count: int = 0,
    lineage_count: int = 0,
    last_signal_at: datetime | None = None,
) -> PortfolioAsset:
    return PortfolioAsset(
        id=uuid.uuid4(),
        tech_id=uuid.uuid4(),
        title=title,
        source_url="https://example.com/" + title.replace(" ", "-"),
        category=category,
        image_url=image_url,
        trust_score=0.75,
        kept_since=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        last_signal_at=last_signal_at,
        signal_count=signal_count,
        lineage_count=lineage_count,
    )


def _empty_view() -> PortfolioView:
    return PortfolioView(active=[], quiet=[], category=None, sort="recency")


def _client_with_portfolio(
    monkeypatch,
    view: PortfolioView,
    capture: list | None = None,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    """Build a TestClient whose portfolio route returns ``view`` without DB access."""
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_portfolio(session, *, category=None, sort="recency", cursor=None):
        if capture is not None:
            capture.append({"category": category, "sort": sort, "cursor": cursor})
        return view

    monkeypatch.setattr("argos.web.app.fetch_portfolio", _fake_fetch_portfolio)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


# ------------------------------------------------------------------ #
# Test 1 — Empty state
# ------------------------------------------------------------------ #

def test_get_portfolio_returns_200_with_empty_view(monkeypatch):
    """Mocked fetch_portfolio returns empty PortfolioView → status 200, empty-state message."""
    client = _client_with_portfolio(monkeypatch, _empty_view())
    resp = client.get("/portfolio")
    assert resp.status_code == 200
    body = resp.text
    assert "아직 Keep한 자산이 없습니다" in body


def test_portfolio_cards_link_to_in_app_reader(monkeypatch):
    """Portfolio cards must open the in-app reader keyed on the tech_item id
    (ARG-138), not bounce to the external source_url, and not mistakenly use
    the user_asset id."""
    asset = _asset(title="Linkable", signal_count=1)
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert f'href="/item/{asset.tech_id}"' in body
    # The user_asset id is NOT a valid /item/{id} key.
    assert f'href="/item/{asset.id}"' not in body
    # Cards no longer point straight at the external source.
    assert f'href="{asset.source_url}"' not in body


def test_portfolio_cards_render_untrack_control(monkeypatch):
    """The Untrack action endpoint must be reachable from the portfolio render
    (ARG-139). Regression guard for the control living in an unincluded
    partial; the button targets its own card by user_asset id."""
    asset = _asset(title="Untrackable", signal_count=1)
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert f'hx-post="/assets/{asset.id}/untrack"' in body
    assert f'id="portfolio-card-{asset.id}"' in body


# ------------------------------------------------------------------ #
# Test 2 — Section ordering
# ------------------------------------------------------------------ #

def test_get_portfolio_renders_active_section_above_quiet(monkeypatch):
    """Assets with signals appear in '새 신호 있음' above '조용함'."""
    active_asset = _asset(title="ActiveAsset", signal_count=2)
    quiet_asset = _asset(title="QuietAsset", signal_count=0)
    view = PortfolioView(
        active=[active_asset],
        quiet=[quiet_asset],
        category=None,
        sort="recency",
    )
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "새 신호 있음" in body
    assert "조용함" in body
    # Section ordering: '새 신호 있음' must appear before '조용함'
    assert body.index("새 신호 있음") < body.index("조용함")
    assert "ActiveAsset" in body
    assert "QuietAsset" in body


# ------------------------------------------------------------------ #
# Test 3 — Badge suppression
# ------------------------------------------------------------------ #

def test_get_portfolio_renders_badges_only_when_nonzero(monkeypatch):
    """signal_count=3, lineage_count=0 → 🔭 appears, 🧬 does not."""
    asset = _asset(title="BadgeTest", signal_count=3, lineage_count=0)
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "🔭" in body
    assert "3" in body
    assert "🧬" not in body


# ------------------------------------------------------------------ #
# Test 4 — Image vs fallback cover
# ------------------------------------------------------------------ #

def test_get_portfolio_renders_image_when_present_else_source_fallback(monkeypatch):
    """Asset with image_url gets <img src>, asset without gets cover--fallback."""
    asset_with_img = _asset(
        title="WithImage",
        image_url="https://img.example.com/thumb.png",
        category=CategoryType.MAINSTREAM,
    )
    asset_no_img = _asset(
        title="NoImage",
        image_url=None,
        category=CategoryType.MAINSTREAM,
    )
    view = PortfolioView(
        active=[asset_with_img],
        quiet=[asset_no_img],
        category=None,
        sort="recency",
    )
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    # Image rendered as <img src>, not inline CSS
    assert '<img class="cover__img" src="https://img.example.com/thumb.png"' in body
    assert "background-image: url(" not in body
    # Fallback class for no-image asset
    assert "cover--fallback" in body
    assert "cover--mainstream" in body


# ------------------------------------------------------------------ #
# Test 5 — Category filter forwarded to service
# ------------------------------------------------------------------ #

def test_get_portfolio_filter_passes_category_to_service(monkeypatch):
    """?category=Mainstream → fetch_portfolio called with category='Mainstream'."""
    capture: list = []
    client = _client_with_portfolio(monkeypatch, _empty_view(), capture=capture)
    resp = client.get("/portfolio?category=Mainstream")
    assert resp.status_code == 200
    assert capture and capture[-1]["category"] == "Mainstream"


# ------------------------------------------------------------------ #
# Test 6 — Sort forwarded to service
# ------------------------------------------------------------------ #

def test_get_portfolio_sort_passes_sort_to_service(monkeypatch):
    """?sort=trust → fetch_portfolio called with sort='trust'."""
    capture: list = []
    client = _client_with_portfolio(monkeypatch, _empty_view(), capture=capture)
    resp = client.get("/portfolio?sort=trust")
    assert resp.status_code == 200
    assert capture and capture[-1]["sort"] == "trust"


# ------------------------------------------------------------------ #
# Test 7 — Invalid category normalizes to None
# ------------------------------------------------------------------ #

def test_get_portfolio_invalid_category_normalizes_to_none(monkeypatch):
    """?category=Garbage → fetch_portfolio called with category=None, status 200."""
    capture: list = []
    client = _client_with_portfolio(monkeypatch, _empty_view(), capture=capture)
    resp = client.get("/portfolio?category=Garbage")
    assert resp.status_code == 200
    assert capture and capture[-1]["category"] is None


# ------------------------------------------------------------------ #
# Test 8 — Invalid sort normalizes to 'recency'
# ------------------------------------------------------------------ #

def test_get_portfolio_invalid_sort_normalizes_to_recency(monkeypatch):
    """?sort=banana → fetch_portfolio called with sort='recency', status 200."""
    capture: list = []
    client = _client_with_portfolio(monkeypatch, _empty_view(), capture=capture)
    resp = client.get("/portfolio?sort=banana")
    assert resp.status_code == 200
    assert capture and capture[-1]["sort"] == "recency"


# ------------------------------------------------------------------ #
# Test 9 — ValueError → 400
# ------------------------------------------------------------------ #

def test_get_portfolio_value_error_returns_400_not_500(monkeypatch):
    """fetch_portfolio raising ValueError → route returns 400, not 500."""
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _raising_fetch(session, *, category=None, sort="recency", cursor=None):
        raise ValueError("bad portfolio query")

    monkeypatch.setattr("argos.web.app.fetch_portfolio", _raising_fetch)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/portfolio")
    assert resp.status_code == 400


# ------------------------------------------------------------------ #
# Test 10 — Full page layout (bonus)
# ------------------------------------------------------------------ #

def test_get_portfolio_renders_full_page_layout(monkeypatch):
    """GET /portfolio renders the base layout with doctype and tabbar."""
    client = _client_with_portfolio(monkeypatch, _empty_view())
    body = client.get("/portfolio").text
    assert "<!DOCTYPE html>" in body
    assert 'class="tabbar"' in body
    assert "포트폴리오" in body


# ------------------------------------------------------------------ #
# Test 11 — kept_since date rendered
# ------------------------------------------------------------------ #

def test_get_portfolio_renders_kept_since_date(monkeypatch):
    """kept_since date is rendered on each card."""
    asset = _asset(title="DateAsset", signal_count=0, lineage_count=0)
    view = PortfolioView(active=[], quiet=[asset], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "Keep 2026-05-01" in body


# ------------------------------------------------------------------ #
# Test 12 — last_signal_at rendered only when present
# ------------------------------------------------------------------ #

def test_get_portfolio_last_signal_at_rendered_when_present(monkeypatch):
    """last_signal_at is rendered when set; omitted when None."""
    with_signal = _asset(
        title="WithSignal",
        signal_count=1,
        last_signal_at=datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc),
    )
    without_signal = _asset(title="WithoutSignal", signal_count=0, last_signal_at=None)
    view = PortfolioView(
        active=[with_signal],
        quiet=[without_signal],
        category=None,
        sort="recency",
    )
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "마지막 신호 2026-06-10" in body
    # Without signal: '마지막 신호' text should appear once (for with_signal only)
    assert body.count("마지막 신호") == 1


# ------------------------------------------------------------------ #
# Test 13 — /portfolio/items fragment route + cursor pagination (ARG-187)
# ------------------------------------------------------------------ #

def _view(active=None, quiet=None, *, category=None, sort="recency", next_cursor=None):
    return PortfolioView(
        active=active or [],
        quiet=quiet or [],
        category=category,
        sort=sort,
        next_cursor=next_cursor,
    )


def test_portfolio_renders_load_more_when_next_cursor(monkeypatch):
    view = _view(quiet=[_asset(title="X")], next_cursor="PCURSOR1")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "load-more" in body
    assert "PCURSOR1" in body
    assert "/portfolio/items" in body


def test_portfolio_no_load_more_when_no_next_cursor(monkeypatch):
    view = _view(quiet=[_asset(title="X")], next_cursor=None)
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "load-more" not in body


def test_portfolio_items_fragment_is_partial_only(monkeypatch):
    view = _view(quiet=[_asset(title="FragAsset")])
    client = _client_with_portfolio(monkeypatch, view)
    resp = client.get("/portfolio/items")
    assert resp.status_code == 200
    body = resp.text
    assert "FragAsset" in body
    assert "<!DOCTYPE html>" not in body
    assert 'class="tabbar"' not in body


def test_portfolio_items_fragment_carries_sort_and_category_in_load_more(monkeypatch):
    view = _view(quiet=[_asset(title="X")], next_cursor="NEXTP", sort="trust")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio/items?sort=trust&category=Alpha").text
    assert "NEXTP" in body
    assert "sort=trust" in body
    assert "category=Alpha" in body


def test_portfolio_items_fragment_passes_cursor_to_service(monkeypatch):
    capture: list = []
    client = _client_with_portfolio(monkeypatch, _empty_view(), capture=capture)
    client.get("/portfolio/items?cursor=ABC")
    assert capture[-1]["cursor"] == "ABC"


def test_portfolio_passes_cursor_to_service(monkeypatch):
    capture: list = []
    client = _client_with_portfolio(monkeypatch, _empty_view(), capture=capture)
    client.get("/portfolio?cursor=XYZ")
    assert capture[-1]["cursor"] == "XYZ"


def _client_real_portfolio() -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session
    return TestClient(app, raise_server_exceptions=False)


def test_portfolio_malformed_cursor_returns_400_not_500():
    client = _client_real_portfolio()
    resp = client.get("/portfolio?cursor=not-a-valid-cursor")
    assert resp.status_code == 400


def test_portfolio_items_malformed_cursor_returns_400_not_500():
    client = _client_real_portfolio()
    resp = client.get("/portfolio/items?cursor=%%%bogus%%%")
    assert resp.status_code == 400


# ------------------------------------------------------------------ #
# ARG-209 — succession handoff banner
# ------------------------------------------------------------------ #

def test_portfolio_renders_handoff_banner_for_replace_successor(monkeypatch):
    """A Keep asset with a Replace successor (lineage_count > 0) gets a
    card-top handoff banner + 이어받기 button, visible before the timeline
    accordion is ever expanded (AC1/AC3)."""
    from argos.web.services.timeline import ReplaceSuccessor

    asset = _asset(title="OldModel", lineage_count=1)
    successor_id = uuid.uuid4()

    async def _fake_replace_successors(session, tech_id):
        assert tech_id == asset.tech_id
        return [ReplaceSuccessor(tech_id=successor_id, title="NewModel")]

    monkeypatch.setattr(
        "argos.web.app.replace_successors", _fake_replace_successors
    )
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "handoff-banner" in body
    assert "NewModel" in body
    assert "이어받기" in body
    assert (
        f'hx-post="/assets/{asset.id}/handoff?successor_tech_id={successor_id}"'
        in body
    )


def test_portfolio_omits_handoff_banner_when_lineage_count_zero(monkeypatch):
    """No succession link at all (lineage_count=0) must skip the
    replace_successors lookup entirely — no extra query for the common case,
    and no banner rendered."""
    asset = _asset(title="Plain", lineage_count=0)

    async def _fake_replace_successors(session, tech_id):
        raise AssertionError("replace_successors must not run for lineage_count=0")

    monkeypatch.setattr(
        "argos.web.app.replace_successors", _fake_replace_successors
    )
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "handoff-banner" not in body


def test_portfolio_omits_handoff_banner_when_only_enhance_or_fork(monkeypatch):
    """lineage_count > 0 but every succession is Enhance/Fork (no Replace) —
    replace_successors returns [] and no banner renders (AC3: those get the
    timeline "이 기술도 Keep" button instead, not a banner)."""
    asset = _asset(title="EnhancedOnly", lineage_count=1)

    async def _fake_replace_successors(session, tech_id):
        return []

    monkeypatch.setattr(
        "argos.web.app.replace_successors", _fake_replace_successors
    )
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    client = _client_with_portfolio(monkeypatch, view)
    body = client.get("/portfolio").text
    assert "handoff-banner" not in body
