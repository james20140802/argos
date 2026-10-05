"""A detail page opened from the portfolio belongs to the portfolio.

The item URL stays the single canonical ``/item/<id>``; where the reader came
from is read from the same-origin Referer. No live database (release CI).
"""
from __future__ import annotations

import re

from tests.web.test_item_detail_route import _client_with_detail, _view
from tests.web.test_portfolio_route import _asset, _client_with_portfolio
from argos.web.services.portfolio import PortfolioView


def _back_link(body: str) -> str:
    return re.search(r'<a class="back-link"[^>]*>.*?</a>', body, re.S).group(0)


def _current_tabs(body: str) -> set[str]:
    return set(re.findall(r'<a class="(?:rail__tab|tab)" href="([^"]+)" aria-current="page"', body))


def test_item_opened_from_portfolio_goes_back_to_portfolio(monkeypatch):
    view = _view()
    client = _client_with_detail(monkeypatch, view)
    resp = client.get(f"/item/{view.id}", headers={"Referer": "http://testserver/portfolio?sort=signal"})
    back = _back_link(resp.text)
    assert 'href="/portfolio"' in back and "포트폴리오" in back
    assert _current_tabs(resp.text) == {"/portfolio"}
    assert "Referer" in resp.headers["vary"]


def test_item_opened_from_feed_or_directly_goes_back_to_feed(monkeypatch):
    view = _view()
    client = _client_with_detail(monkeypatch, view)
    for headers in ({"Referer": "http://testserver/feed"}, {}):
        resp = client.get(f"/item/{view.id}", headers=headers)
        back = _back_link(resp.text)
        assert 'href="/feed"' in back and "관측 피드" in back
        assert _current_tabs(resp.text) == {"/feed"}


def test_foreign_referer_is_not_trusted_as_portfolio(monkeypatch):
    view = _view()
    client = _client_with_detail(monkeypatch, view)
    resp = client.get(f"/item/{view.id}", headers={"Referer": "https://evil.example/portfolio"})
    assert 'href="/feed"' in _back_link(resp.text)


def test_portfolio_cover_morphs_into_the_item_hero(monkeypatch):
    asset = _asset(title="Morph", image_url="https://example.com/og.png")
    view = PortfolioView(active=[asset], quiet=[], category=None, sort="recency")
    body = _client_with_portfolio(monkeypatch, view).get("/portfolio").text
    # Same name the item page gives its hero (c<tech_id hex>).
    assert f'data-vt="c{asset.tech_id.hex}"' in body
