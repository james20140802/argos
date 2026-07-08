"""Tests for src/argos/crawler/rss_fetcher.py (ARG-52)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.config import RSSFeedConfig
from argos.crawler.rss_fetcher import (
    _truncate,
    fetch_rss_feed,
    run_rss_fetchers,
)
from argos.models.tech_item import CategoryType

# ---------------------------------------------------------------------------
# Minimal fake feedparser results
# ---------------------------------------------------------------------------

_MAINSTREAM_ENTRY = SimpleNamespace(
    title="GPT-5 Is Here",
    link="https://openai.com/blog/gpt-5",
    summary="OpenAI announces GPT-5 with groundbreaking capabilities.",
    content=None,
    description=None,
)

_REDDIT_ENTRY = SimpleNamespace(
    title="LocalLLaMA weekly megathread",
    link="https://www.reddit.com/r/LocalLLaMA/comments/abc123/",
    summary="Community discussion about running LLMs locally.",
    content=None,
    description=None,
)

_ENTRY_NO_LINK = SimpleNamespace(
    title="Article with no link",
    link=None,
    summary="...",
    content=None,
    description=None,
)

_ENTRY_NO_TITLE = SimpleNamespace(
    title=None,
    link="https://example.com/no-title",
    summary="...",
    content=None,
    description=None,
)


def _make_parsed(entries, status=200):
    parsed = MagicMock()
    parsed.entries = entries
    parsed.status = status
    return parsed


# ---------------------------------------------------------------------------
# Fixtures: robots patch (allow by default for rss_fetcher module)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def allow_robots(monkeypatch):
    """Patch is_robots_allowed at the rss_fetcher import site to return True."""
    monkeypatch.setattr(
        "argos.crawler.rss_fetcher.is_robots_allowed",
        AsyncMock(return_value=True),
    )


# ---------------------------------------------------------------------------
# _truncate unit tests
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    text = "Hello, world!"
    assert _truncate(text) == text


def test_truncate_long_string_at_byte_boundary():
    # Create a string whose UTF-8 length exceeds 8 KB
    long_text = "x" * (8 * 1024 + 100)
    result = _truncate(long_text)
    assert len(result.encode("utf-8")) <= 8 * 1024


def test_truncate_multibyte_does_not_split_codepoint():
    # Korean characters are 3 bytes each in UTF-8.
    # Build a string that is just over the limit.
    char = "가"  # 3 bytes
    count = (8 * 1024 // 3) + 10  # deliberately over limit
    text = char * count
    result = _truncate(text)
    assert len(result.encode("utf-8")) <= 8 * 1024
    # result must be valid (no truncation mid-codepoint errors)
    result.encode("utf-8")  # would raise if broken


# ---------------------------------------------------------------------------
# fetch_rss_feed: parses entries into expected dict shape
# ---------------------------------------------------------------------------


async def test_fetch_rss_feed_returns_expected_shape():
    parsed = _make_parsed([_MAINSTREAM_ENTRY])
    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=AsyncMock(return_value=parsed)):
        items = await fetch_rss_feed(
            "https://openai.com/blog/rss.xml", CategoryType.MAINSTREAM
        )
    assert len(items) == 1
    item = items[0]
    assert item["title"] == "GPT-5 Is Here"
    assert item["source_url"] == "https://openai.com/blog/gpt-5"
    assert "GPT-5 Is Here" in item["raw_content"]
    assert item["_source_category"] is CategoryType.MAINSTREAM


async def test_fetch_rss_feed_returns_alpha_category_for_reddit():
    parsed = _make_parsed([_REDDIT_ENTRY])
    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=AsyncMock(return_value=parsed)):
        items = await fetch_rss_feed(
            "https://www.reddit.com/r/LocalLLaMA/.rss", CategoryType.ALPHA
        )
    assert len(items) == 1
    assert items[0]["_source_category"] is CategoryType.ALPHA


# ---------------------------------------------------------------------------
# fetch_rss_feed: skips entries with missing link or title
# ---------------------------------------------------------------------------


async def test_fetch_rss_feed_skips_entry_missing_link():
    parsed = _make_parsed([_ENTRY_NO_LINK, _MAINSTREAM_ENTRY])
    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=AsyncMock(return_value=parsed)):
        items = await fetch_rss_feed(
            "https://openai.com/blog/rss.xml", CategoryType.MAINSTREAM
        )
    assert len(items) == 1
    assert items[0]["title"] == "GPT-5 Is Here"


async def test_fetch_rss_feed_skips_entry_missing_title():
    parsed = _make_parsed([_ENTRY_NO_TITLE, _REDDIT_ENTRY])
    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=AsyncMock(return_value=parsed)):
        items = await fetch_rss_feed(
            "https://www.reddit.com/r/LocalLLaMA/.rss", CategoryType.ALPHA
        )
    assert len(items) == 1
    assert items[0]["title"] == "LocalLLaMA weekly megathread"


# ---------------------------------------------------------------------------
# fetch_rss_feed: swallows feedparser errors on broken feeds
# ---------------------------------------------------------------------------


async def test_fetch_rss_feed_swallows_feedparser_exception():
    with patch(
        "argos.crawler.rss_fetcher.asyncio.to_thread",
        new=AsyncMock(side_effect=RuntimeError("connection reset")),
    ):
        items = await fetch_rss_feed(
            "https://broken.example.com/rss.xml", CategoryType.MAINSTREAM
        )
    assert items == []


async def test_fetch_rss_feed_returns_empty_on_http_error_status():
    parsed = _make_parsed([], status=404)
    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=AsyncMock(return_value=parsed)):
        items = await fetch_rss_feed(
            "https://openai.com/blog/rss.xml", CategoryType.MAINSTREAM
        )
    assert items == []


# ---------------------------------------------------------------------------
# fetch_rss_feed: respects robots disallow
# ---------------------------------------------------------------------------


async def test_fetch_rss_feed_respects_robots_disallow(monkeypatch):
    monkeypatch.setattr(
        "argos.crawler.rss_fetcher.is_robots_allowed",
        AsyncMock(return_value=False),
    )
    with patch(
        "argos.crawler.rss_fetcher.asyncio.to_thread",
        new=AsyncMock(side_effect=AssertionError("should not be called")),
    ):
        items = await fetch_rss_feed(
            "https://disallowed.example.com/rss.xml", CategoryType.MAINSTREAM
        )
    assert items == []


# ---------------------------------------------------------------------------
# fetch_rss_feed: truncates oversized entries to 8 KB
# ---------------------------------------------------------------------------


async def test_fetch_rss_feed_truncates_large_content():
    big_entry = SimpleNamespace(
        title="Big Post",
        link="https://example.com/big",
        summary="x" * (16 * 1024),  # 16 KB — twice the limit
        content=None,
        description=None,
    )
    parsed = _make_parsed([big_entry])
    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=AsyncMock(return_value=parsed)):
        items = await fetch_rss_feed(
            "https://example.com/rss.xml", CategoryType.MAINSTREAM
        )
    assert len(items) == 1
    assert len(items[0]["raw_content"].encode("utf-8")) <= 8 * 1024


# ---------------------------------------------------------------------------
# run_rss_fetchers: returns combined results from multiple feeds
# ---------------------------------------------------------------------------


async def test_run_rss_fetchers_combines_multiple_feeds():
    feeds = [
        RSSFeedConfig(url="https://openai.com/blog/rss.xml", category="Mainstream"),
        RSSFeedConfig(url="https://www.reddit.com/r/LocalLLaMA/.rss", category="Alpha"),
    ]
    mainstream_parsed = _make_parsed([_MAINSTREAM_ENTRY])
    alpha_parsed = _make_parsed([_REDDIT_ENTRY])

    call_count = 0

    async def fake_to_thread(func, url, agent):
        nonlocal call_count
        call_count += 1
        if "openai" in url:
            return mainstream_parsed
        return alpha_parsed

    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=fake_to_thread):
        items = await run_rss_fetchers(feeds)

    assert len(items) == 2
    cats = {item["_source_category"] for item in items}
    assert CategoryType.MAINSTREAM in cats
    assert CategoryType.ALPHA in cats


async def test_run_rss_fetchers_returns_empty_for_empty_feeds():
    items = await run_rss_fetchers([])
    assert items == []


async def test_run_rss_fetchers_swallows_per_feed_failure():
    """One broken feed must not prevent results from the healthy feed."""
    feeds = [
        RSSFeedConfig(url="https://broken.example.com/rss.xml", category="Mainstream"),
        RSSFeedConfig(url="https://openai.com/blog/rss.xml", category="Mainstream"),
    ]
    mainstream_parsed = _make_parsed([_MAINSTREAM_ENTRY])

    async def fake_to_thread(func, url, agent):
        if "broken" in url:
            raise RuntimeError("connection refused")
        return mainstream_parsed

    with patch("argos.crawler.rss_fetcher.asyncio.to_thread", new=fake_to_thread):
        items = await run_rss_fetchers(feeds)

    assert len(items) == 1
    assert items[0]["title"] == "GPT-5 Is Here"


# ---------------------------------------------------------------------------
# Tests: HTML title cleaning in _entry_to_dict (ARG-129)
# ---------------------------------------------------------------------------


def test_entry_to_dict_cleans_html_entities_in_title() -> None:
    """RSS feed entries may carry HTML entities in their title (ARG-129)."""
    from argos.crawler.rss_fetcher import _entry_to_dict
    from argos.models.tech_item import CategoryType

    entry = SimpleNamespace(
        title="AT&amp;T launches AI &mdash; first look",
        link="https://example.com/att-ai",
        summary="Details here.",
        content=None,
        description=None,
        published_parsed=None,
    )
    result = _entry_to_dict(entry, CategoryType.MAINSTREAM)
    assert result is not None
    assert result["title"] == "AT&T launches AI — first look"
    assert "&amp;" not in result["title"]
    assert "&mdash;" not in result["title"]


def test_entry_to_dict_cleans_html_tags_in_title() -> None:
    """Some RSS feeds include HTML markup in their entry title (ARG-129)."""
    from argos.crawler.rss_fetcher import _entry_to_dict
    from argos.models.tech_item import CategoryType

    entry = SimpleNamespace(
        title="<b>Breaking:</b> New LLM beats GPT-4",
        link="https://example.com/new-llm",
        summary="Details.",
        content=None,
        description=None,
        published_parsed=None,
    )
    result = _entry_to_dict(entry, CategoryType.MAINSTREAM)
    assert result is not None
    assert result["title"] == "Breaking: New LLM beats GPT-4"
    assert "<b>" not in result["title"]
