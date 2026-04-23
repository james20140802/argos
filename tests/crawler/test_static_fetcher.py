from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import respx

from argos.crawler.user_agents import USER_AGENTS, random_user_agent
from argos.crawler.static_fetcher import (
    fetch_github_trending,
    fetch_hackernews_top,
    filter_duplicate_urls,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _github_trending_html() -> str:
    return (FIXTURES_DIR / "github_trending.html").read_text()


def test_random_user_agent_returns_from_list() -> None:
    ua = random_user_agent()
    assert ua in USER_AGENTS


async def test_fetch_github_trending_parses_repos() -> None:
    with respx.mock:
        respx.get("https://github.com/trending").mock(
            return_value=httpx.Response(200, text=_github_trending_html())
        )
        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    assert len(items) == 2
    for item in items:
        assert "title" in item
        assert "source_url" in item
        assert "raw_content" in item
        assert item["source_url"].startswith("https://github.com/")


async def test_fetch_github_trending_sends_user_agent_header() -> None:
    sent_headers: dict = {}

    def capture(request: httpx.Request) -> httpx.Response:
        sent_headers.update(dict(request.headers))
        return httpx.Response(200, text=_github_trending_html())

    with respx.mock:
        respx.get("https://github.com/trending").mock(side_effect=capture)
        async with httpx.AsyncClient() as client:
            await fetch_github_trending(client)

    assert sent_headers.get("user-agent") in USER_AGENTS


async def test_fetch_hackernews_top_returns_items() -> None:
    top_ids = [1, 2, 3]
    items_data = {
        1: {"id": 1, "title": "Story One", "url": "https://example.com/1", "text": ""},
        2: {"id": 2, "title": "Story Two", "url": "https://example.com/2", "text": ""},
        3: {"id": 3, "title": "Story Three", "url": "https://example.com/3", "text": ""},
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        for item_id, data in items_data.items():
            respx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
            ).mock(return_value=httpx.Response(200, text=json.dumps(data)))

        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=3)

    assert len(result) == 3
    for item in result:
        assert item["source_url"].startswith("https://")


async def test_fetch_hackernews_top_handles_malformed_payloads() -> None:
    top_ids = [10, 20, 30]
    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        # 10 → non-JSON body
        respx.get("https://hacker-news.firebaseio.com/v0/item/10.json").mock(
            return_value=httpx.Response(200, text="<html>oops</html>")
        )
        # 20 → JSON but not a dict
        respx.get("https://hacker-news.firebaseio.com/v0/item/20.json").mock(
            return_value=httpx.Response(200, text=json.dumps([1, 2, 3]))
        )
        # 30 → unsafe scheme in url → fall back to canonical HN URL
        respx.get("https://hacker-news.firebaseio.com/v0/item/30.json").mock(
            return_value=httpx.Response(
                200,
                text=json.dumps(
                    {"id": 30, "title": "Hi", "url": "javascript:alert(1)", "text": ""}
                ),
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=3)

    assert len(result) == 1
    assert result[0]["source_url"] == "https://news.ycombinator.com/item?id=30"


async def test_fetch_hackernews_top_returns_empty_on_invalid_topstories_json() -> None:
    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text="<html>error</html>")
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=5)

    assert result == []


async def test_filter_duplicate_urls_removes_existing() -> None:
    existing_url = "https://existing.com/x"
    new_url = "https://new.com/y"

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [existing_url]

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    items = [
        {"title": "Existing", "source_url": existing_url, "raw_content": "old"},
        {"title": "New", "source_url": new_url, "raw_content": "new"},
    ]

    result = await filter_duplicate_urls(mock_session, items)

    assert len(result) == 1
    assert result[0]["source_url"] == new_url
