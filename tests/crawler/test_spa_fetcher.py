from __future__ import annotations

from unittest.mock import patch

import pytest

from argos.config import SPASourceConfig


def _make_config(
    listing_url: str = "https://example.com/news",
    link_pattern: str = r"^/news/[^/]+$",
    base_url: str = "https://example.com",
    max_items: int = 5,
    name: str = "test",
    category: str = "Mainstream",
) -> SPASourceConfig:
    return SPASourceConfig(
        listing_url=listing_url,
        link_pattern=link_pattern,
        base_url=base_url,
        max_items=max_items,
        name=name,
        category=category,
    )


@pytest.mark.asyncio
async def test_fetch_spa_source_returns_empty_when_robots_disallowed():
    from argos.crawler.spa_fetcher import fetch_spa_source

    config = _make_config()
    with patch("argos.crawler.spa_fetcher.is_robots_allowed", return_value=False):
        result = await fetch_spa_source(config)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_spa_source_extracts_matching_links():
    from argos.crawler.spa_fetcher import fetch_spa_source

    html = '''
    <html><body>
      <a href="/news/article-one">Article One</a>
      <a href="/news/article-two">Article Two</a>
      <a href="/about">About (should not match)</a>
    </body></html>
    '''

    config = _make_config(max_items=10)

    async def mock_fetch_dynamic(url):
        return {"title": f"Title for {url}", "source_url": url, "raw_content": "body"}

    with patch("argos.crawler.spa_fetcher.is_robots_allowed", return_value=True):
        with patch("argos.crawler.spa_fetcher._load_page_html", return_value=(html, "https://example.com/news")):
            with patch("argos.crawler.spa_fetcher.fetch_dynamic_page", side_effect=mock_fetch_dynamic):
                result = await fetch_spa_source(config)

    assert len(result) == 2
    urls = {r["source_url"] for r in result}
    assert "https://example.com/news/article-one" in urls
    assert "https://example.com/news/article-two" in urls
    assert all("https://example.com/about" not in r["source_url"] for r in result)


@pytest.mark.asyncio
async def test_fetch_spa_source_respects_max_items():
    from argos.crawler.spa_fetcher import fetch_spa_source

    html = "\n".join(
        f'<a href="/news/article-{i}">Article {i}</a>' for i in range(20)
    )

    config = _make_config(max_items=3)

    async def mock_fetch_dynamic(url):
        return {"title": url, "source_url": url, "raw_content": "body"}

    with patch("argos.crawler.spa_fetcher.is_robots_allowed", return_value=True):
        with patch("argos.crawler.spa_fetcher._load_page_html", return_value=(html, "https://example.com/news")):
            with patch("argos.crawler.spa_fetcher.fetch_dynamic_page", side_effect=mock_fetch_dynamic):
                result = await fetch_spa_source(config)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_fetch_spa_source_deduplicates_links():
    from argos.crawler.spa_fetcher import fetch_spa_source

    html = '''
    <a href="/news/dupe">Dupe</a>
    <a href="/news/dupe">Dupe again</a>
    '''
    config = _make_config(max_items=10)

    async def mock_fetch_dynamic(url):
        return {"title": url, "source_url": url, "raw_content": "body"}

    with patch("argos.crawler.spa_fetcher.is_robots_allowed", return_value=True):
        with patch("argos.crawler.spa_fetcher._load_page_html", return_value=(html, "https://example.com/news")):
            with patch("argos.crawler.spa_fetcher.fetch_dynamic_page", side_effect=mock_fetch_dynamic):
                result = await fetch_spa_source(config)

    assert len(result) == 1


@pytest.mark.asyncio
async def test_fetch_spa_source_stamps_source_and_category():
    from argos.crawler.spa_fetcher import fetch_spa_source
    from argos.models.tech_item import CategoryType

    html = '<a href="/news/article">Article</a>'
    config = _make_config(name="mysite", category="Alpha")

    async def mock_fetch_dynamic(url):
        return {"title": url, "source_url": url, "raw_content": "body"}

    with patch("argos.crawler.spa_fetcher.is_robots_allowed", return_value=True):
        with patch("argos.crawler.spa_fetcher._load_page_html", return_value=(html, "https://example.com/news")):
            with patch("argos.crawler.spa_fetcher.fetch_dynamic_page", side_effect=mock_fetch_dynamic):
                result = await fetch_spa_source(config)

    assert len(result) == 1
    assert result[0]["_source"] == "spa:mysite"
    assert result[0]["_source_category"] == CategoryType.ALPHA


@pytest.mark.asyncio
async def test_run_spa_fetchers_skips_failed_source():
    from argos.crawler.spa_fetcher import run_spa_fetchers

    config_ok = _make_config(name="ok")
    config_fail = _make_config(listing_url="https://broken.example.com/news", name="fail")

    async def side_effect(config):
        if config.name == "fail":
            raise RuntimeError("network error")
        return [{"title": "t", "source_url": "https://ok.example.com/a", "raw_content": "b", "_source": "spa:ok", "_source_category": "Mainstream", "_published_at": None}]

    with patch("argos.crawler.spa_fetcher.fetch_spa_source", side_effect=side_effect):
        result = await run_spa_fetchers([config_ok, config_fail])

    assert len(result) == 1
    assert result[0]["_source"] == "spa:ok"
