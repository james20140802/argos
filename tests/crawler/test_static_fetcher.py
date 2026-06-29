from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from argos.crawler._robots import RobotsDisallowed
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


def _stub_github_readmes_missing() -> None:
    """Register catch-all 404s so README enrichment fetches don't blow up
    tests that don't care about README content."""
    respx.get(url__regex=r"https://raw\.githubusercontent\.com/.*").mock(
        return_value=httpx.Response(404, text="not found")
    )


def _stub_external_article_bodies_missing() -> None:
    """Register catch-all 404s for arbitrary external HN-link hosts so body
    enrichment falls back to title in tests that don't care about it."""
    respx.get(url__regex=r"https?://example\.com/.*").mock(
        return_value=httpx.Response(404, text="not found")
    )
    respx.get(url__regex=r"https?://example\.com/robots\.txt").mock(
        return_value=httpx.Response(404, text="not found")
    )


async def test_fetch_github_trending_parses_repos() -> None:
    with respx.mock:
        respx.get("https://github.com/trending").mock(
            return_value=httpx.Response(200, text=_github_trending_html())
        )
        _stub_github_readmes_missing()
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
        _stub_github_readmes_missing()
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
        _stub_external_article_bodies_missing()

        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=3)

    assert len(result) == 3
    for item in result:
        assert item["source_url"].startswith("https://")


async def test_fetch_hackernews_top_passes_real_robots_check(monkeypatch) -> None:
    """Regression for issue #5: HN's host has a generic Disallow-all
    robots.txt, but it's a vendor-published public API. With the real
    is_robots_allowed wired in, fetch_hackernews_top must still return items
    *without* any /robots.txt route registered (the allowlist short-circuits
    the fetch)."""
    from argos.crawler import _robots
    from argos.crawler import static_fetcher

    monkeypatch.setattr(static_fetcher, "is_robots_allowed", _robots.is_robots_allowed)
    _robots._robots_cache.clear()
    _robots._robots_origin_locks.clear()

    top_ids = [1, 2]
    items_data = {
        1: {"id": 1, "title": "S1", "url": "https://example.com/1", "text": ""},
        2: {"id": 2, "title": "S2", "url": "https://example.com/2", "text": ""},
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        for item_id, data in items_data.items():
            respx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
            ).mock(return_value=httpx.Response(200, text=json.dumps(data)))
        _stub_external_article_bodies_missing()

        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=2)

    assert len(result) == 2
    assert {r["title"] for r in result} == {"S1", "S2"}


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


async def test_fetch_github_trending_retries_on_transient_5xx(monkeypatch) -> None:
    """Static fetcher must retry on 503 before giving up."""
    from unittest.mock import AsyncMock

    sleep_mock = AsyncMock()
    monkeypatch.setattr("argos.crawler.static_fetcher.asyncio.sleep", sleep_mock)

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text=_github_trending_html())

    with respx.mock:
        respx.get("https://github.com/trending").mock(side_effect=handler)
        _stub_github_readmes_missing()
        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    assert call_count["n"] == 3
    assert len(items) == 2
    assert sleep_mock.await_count >= 1


async def test_fetch_github_trending_gives_up_after_max_attempts(monkeypatch) -> None:
    """After exhausting retries, a 503 must raise rather than silently succeed."""
    from unittest.mock import AsyncMock

    import pytest

    monkeypatch.setattr(
        "argos.crawler.static_fetcher.asyncio.sleep", AsyncMock()
    )

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(503, text="unavailable")

    with respx.mock:
        respx.get("https://github.com/trending").mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_github_trending(client)

    assert call_count["n"] == 3


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


async def test_get_with_retry_raises_when_robots_disallows(monkeypatch) -> None:
    """No content GET is issued when robots.txt disallows the URL."""
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.is_robots_allowed",
        AsyncMock(return_value=False),
    )

    with respx.mock:
        # If a GET were made, respx would raise an error — so the absence of
        # any route registration proves no content request was issued.
        async with httpx.AsyncClient() as client:
            with pytest.raises(RobotsDisallowed):
                await fetch_github_trending(client)


async def test_get_with_retry_retries_on_429(monkeypatch) -> None:
    """429 is in _RETRYABLE_STATUS_CODES and must trigger a retry."""
    monkeypatch.setattr("argos.crawler.static_fetcher.asyncio.sleep", AsyncMock())
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, text=_github_trending_html())

    with respx.mock:
        respx.get("https://github.com/trending").mock(side_effect=handler)
        _stub_github_readmes_missing()
        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    assert call_count["n"] == 2
    assert len(items) == 2


async def test_filter_duplicate_urls_returns_empty_on_empty_input() -> None:
    session = AsyncMock()
    result = await filter_duplicate_urls(session, [])
    assert result == []
    session.execute.assert_not_called()


async def test_filter_duplicate_urls_removes_all_when_all_duplicates() -> None:
    url = "https://existing.com/x"
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [url]
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    items = [{"title": "Existing", "source_url": url, "raw_content": "c"}]
    result = await filter_duplicate_urls(mock_session, items)
    assert result == []


async def test_filter_duplicate_urls_deduplicates_within_batch() -> None:
    """Two items with the same URL in one batch — only the first survives."""
    url = "https://new.com/y"
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    items = [
        {"title": "first", "source_url": url, "raw_content": "a"},
        {"title": "second", "source_url": url, "raw_content": "b"},
    ]
    result = await filter_duplicate_urls(mock_session, items)
    assert len(result) == 1
    assert result[0]["title"] == "first"


async def test_fetch_github_trending_merges_readme_into_raw_content() -> None:
    readme_body = "# repo1\n\nA full README with code examples and benchmarks."
    with respx.mock:
        respx.get("https://github.com/trending").mock(
            return_value=httpx.Response(200, text=_github_trending_html())
        )
        respx.get(
            "https://raw.githubusercontent.com/owner1/repo1/HEAD/README.md"
        ).mock(return_value=httpx.Response(200, text=readme_body))
        respx.get(
            "https://raw.githubusercontent.com/owner2/repo2/HEAD/README.md"
        ).mock(return_value=httpx.Response(404, text="not found"))
        respx.get(
            "https://raw.githubusercontent.com/owner2/repo2/HEAD/README.rst"
        ).mock(return_value=httpx.Response(404, text="not found"))
        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    by_url = {i["source_url"]: i for i in items}
    enriched = by_url["https://github.com/owner1/repo1"]
    plain = by_url["https://github.com/owner2/repo2"]
    assert "code examples and benchmarks" in enriched["raw_content"]
    assert enriched["raw_content"].startswith("A cool repository")
    assert plain["raw_content"] == "Another awesome repository."


async def test_fetch_github_trending_falls_back_to_rst_readme() -> None:
    rst_body = "repo1 README written in reStructuredText."
    with respx.mock:
        respx.get("https://github.com/trending").mock(
            return_value=httpx.Response(200, text=_github_trending_html())
        )
        respx.get(
            "https://raw.githubusercontent.com/owner1/repo1/HEAD/README.md"
        ).mock(return_value=httpx.Response(404, text="not found"))
        respx.get(
            "https://raw.githubusercontent.com/owner1/repo1/HEAD/README.rst"
        ).mock(return_value=httpx.Response(200, text=rst_body))
        respx.get(
            url__regex=r"https://raw\.githubusercontent\.com/owner2/.*"
        ).mock(return_value=httpx.Response(404, text="not found"))
        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    enriched = next(i for i in items if i["source_url"].endswith("/owner1/repo1"))
    assert "reStructuredText" in enriched["raw_content"]


async def test_fetch_github_trending_truncates_oversized_readme() -> None:
    huge_readme = "x" * 20_000
    with respx.mock:
        respx.get("https://github.com/trending").mock(
            return_value=httpx.Response(200, text=_github_trending_html())
        )
        respx.get(
            url__regex=r"https://raw\.githubusercontent\.com/.*/README\.md"
        ).mock(return_value=httpx.Response(200, text=huge_readme))
        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    for item in items:
        assert len(item["raw_content"].encode("utf-8")) <= 8 * 1024


async def test_fetch_hackernews_top_enriches_external_link_with_body() -> None:
    article_html = """
        <html><body><article>
            <h1>Headline</h1>
            <p>Detailed analysis with benchmarks, code samples, and prior art comparison.</p>
        </article></body></html>
    """
    top_ids = [42]
    item_payload = {
        "id": 42,
        "title": "Cool announcement",
        "url": "https://example.com/post",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/42.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://example.com/post").mock(
            return_value=httpx.Response(
                200, text=article_html, headers={"content-type": "text/html"}
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert "benchmarks" in result[0]["raw_content"]
    assert result[0]["raw_content"].startswith("Cool announcement")


async def test_fetch_hackernews_top_falls_back_to_title_when_body_fetch_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.asyncio.sleep", AsyncMock()
    )
    top_ids = [99]
    item_payload = {
        "id": 99,
        "title": "Title only",
        "url": "https://broken.example.com/x",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/99.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://broken.example.com/x").mock(
            return_value=httpx.Response(500, text="boom")
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["raw_content"] == "Title only"


async def test_fetch_hackernews_top_skips_body_fetch_for_unsafe_url(monkeypatch) -> None:
    """SSRF guard: an HN story whose `url` resolves to a private/internal
    host must not be fetched for body enrichment — fall back to title."""
    from argos.crawler import static_fetcher

    monkeypatch.setattr(
        static_fetcher, "_is_safe_url", AsyncMock(return_value=False)
    )

    top_ids = [1]
    item_payload = {
        "id": 1,
        "title": "Suspicious",
        "url": "http://internal.host/secret",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/1.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        # No mock for internal.host — if a fetch happens, respx raises.
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["raw_content"] == "Suspicious"


async def test_fetch_hackernews_top_keeps_text_field_for_ask_hn() -> None:
    """Ask HN-style stories with a `text` field skip the body fetch entirely."""
    top_ids = [7]
    item_payload = {
        "id": 7,
        "title": "Ask HN: foo?",
        "url": None,
        "text": "Body provided in the HN post itself.",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/7.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["source_url"] == "https://news.ycombinator.com/item?id=7"
    assert "Body provided" in result[0]["raw_content"]


# ---------------------------------------------------------------------------
# Tests: _published_at for fetch_github_trending
# ---------------------------------------------------------------------------

async def test_fetch_github_trending_published_at_is_none(monkeypatch) -> None:
    """GitHub Trending items always have _published_at=None.

    Trending freshness is the crawl/discovery date, not the repo creation date,
    so the briefing query's COALESCE fallback to DB created_at is correct.
    """
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.is_robots_allowed",
        AsyncMock(return_value=True),
    )

    trending_html = """<html><body>
    <article class="Box-row">
      <h2 class="h3 lh-condensed"><a href="/owner/myrepo">owner / myrepo</a></h2>
      <p class="col-9 color-fg-muted">A cool repo</p>
    </article>
    </body></html>"""

    with respx.mock:
        respx.get("https://github.com/trending").mock(
            return_value=httpx.Response(200, text=trending_html)
        )
        respx.get("https://raw.githubusercontent.com/owner/myrepo/HEAD/README.md").mock(
            return_value=httpx.Response(200, text="# MyRepo\nA cool project.")
        )
        respx.get("https://raw.githubusercontent.com/owner/myrepo/HEAD/README.rst").mock(
            return_value=httpx.Response(404, text="")
        )

        async with httpx.AsyncClient() as client:
            items = await fetch_github_trending(client)

    assert len(items) == 1
    assert items[0]["_published_at"] is None


# ---------------------------------------------------------------------------
# Tests: HTML title cleaning (ARG-129)
# ---------------------------------------------------------------------------

async def test_fetch_hackernews_top_cleans_html_entities_in_title() -> None:
    """HN titles with HTML entities must be decoded before storage (ARG-129)."""
    top_ids = [55]
    item_payload = {
        "id": 55,
        "title": "Ask HN: Is Rust&#x2F;Go the right choice?",
        "url": None,
        "text": "",
    }
    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/55.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["title"] == "Ask HN: Is Rust/Go the right choice?"
    assert "&#x2F;" not in result[0]["title"]


async def test_fetch_hackernews_top_cleans_html_tags_in_title() -> None:
    """HN titles with HTML tags must be stripped before storage (ARG-129)."""
    top_ids = [56]
    item_payload = {
        "id": 56,
        "title": "Ask HN: <b>bold</b> &amp; friends",
        "url": None,
        "text": "<p>Some <i>body text</i> here</p>",
    }
    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/56.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["title"] == "Ask HN: bold & friends"
    assert "<b>" not in result[0]["title"]
    assert "&amp;" not in result[0]["title"]


async def test_fetch_hackernews_top_text_html_does_not_contaminate_derived_title() -> None:
    """HN text-post HTML must not bleed into the first line of raw_content.

    save_node derives TechItem.title from the first non-empty line of raw_text
    (= raw_content).  If the HN `text` field contains HTML and is joined onto
    the same line as title, the stored title picks up the markup.
    """
    top_ids = [57]
    item_payload = {
        "id": 57,
        "title": "Ask HN: thoughts on DeepSeek?",
        "url": None,
        "text": "<p><i>DeepSeek</i> just released a new model with <b>crazy</b> benchmarks.",
    }
    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/57.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    item = result[0]
    # The title field must be clean
    assert item["title"] == "Ask HN: thoughts on DeepSeek?"
    # The first line of raw_content is what save_node stores as TechItem.title —
    # it must not contain HTML tags from the text body.
    first_line = item["raw_content"].splitlines()[0]
    assert first_line == "Ask HN: thoughts on DeepSeek?"
    assert "<p>" not in first_line
    assert "<i>" not in first_line
    # The text body content should still appear in raw_content (after the newline)
    assert "DeepSeek" in item["raw_content"]


# ──────────────────────────────────────────────────────────────────────────
# ARG-150: og:image extraction during external article body enrichment
# ──────────────────────────────────────────────────────────────────────────


async def test_fetch_hackernews_top_extracts_og_image_from_external_article() -> None:
    article_html = """
        <html><head>
            <meta property="og:image" content="https://cdn.example.com/cover.jpg">
        </head><body><article>
            <h1>Headline</h1>
            <p>Detailed analysis.</p>
        </article></body></html>
    """
    top_ids = [42]
    item_payload = {
        "id": 42,
        "title": "Cool announcement",
        "url": "https://example.com/post",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/42.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://example.com/post").mock(
            return_value=httpx.Response(
                200, text=article_html, headers={"content-type": "text/html"}
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["image_url"] == "https://cdn.example.com/cover.jpg"


async def test_fetch_hackernews_top_image_url_is_none_when_no_og_image() -> None:
    """After ARG-177: no og/twitter/body image falls back to the domain favicon."""
    article_html = (
        "<html><body><article><h1>Headline</h1>"
        "<p>No og:image here.</p></article></body></html>"
    )
    top_ids = [43]
    item_payload = {
        "id": 43,
        "title": "Plain article",
        "url": "https://example.com/plain",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/43.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://example.com/plain").mock(
            return_value=httpx.Response(
                200, text=article_html, headers={"content-type": "text/html"}
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    # Favicon fallback — not None any more (ARG-177)
    assert result[0].get("image_url") == "https://example.com/favicon.ico"


async def test_fetch_hackernews_top_image_url_falls_back_to_twitter_image() -> None:
    article_html = (
        '<html><head>'
        '<meta name="twitter:image" content="https://cdn.example.com/tw.jpg">'
        '</head><body><article><p>Body.</p></article></body></html>'
    )
    top_ids = [44]
    item_payload = {
        "id": 44,
        "title": "Twitter image only",
        "url": "https://example.com/tw",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/44.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://example.com/tw").mock(
            return_value=httpx.Response(
                200, text=article_html, headers={"content-type": "text/html"}
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0]["image_url"] == "https://cdn.example.com/tw.jpg"


async def test_fetch_hackernews_top_image_url_rejects_data_uri(monkeypatch) -> None:
    """Pipeline must not raise on an invalid (data:) og:image — falls back to favicon (ARG-177)."""
    article_html = (
        '<html><head>'
        '<meta property="og:image" content="data:image/png;base64,abc">'
        '</head><body><article><p>Body.</p></article></body></html>'
    )
    top_ids = [45]
    item_payload = {
        "id": 45,
        "title": "Bad og:image",
        "url": "https://example.com/bad",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/45.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://example.com/bad").mock(
            return_value=httpx.Response(
                200, text=article_html, headers={"content-type": "text/html"}
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    # Invalid og:image → body image absent → favicon fallback (ARG-177)
    assert result[0].get("image_url") == "https://example.com/favicon.ico"


async def test_fetch_hackernews_top_image_url_none_when_body_fetch_fails(
    monkeypatch,
) -> None:
    """When the external body fetch fails entirely, image_url must be None."""
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.asyncio.sleep", AsyncMock()
    )
    top_ids = [46]
    item_payload = {
        "id": 46,
        "title": "Body failure",
        "url": "https://broken.example.com/x",
        "text": "",
    }

    with respx.mock:
        respx.get("https://hacker-news.firebaseio.com/v0/topstories.json").mock(
            return_value=httpx.Response(200, text=json.dumps(top_ids))
        )
        respx.get("https://hacker-news.firebaseio.com/v0/item/46.json").mock(
            return_value=httpx.Response(200, text=json.dumps(item_payload))
        )
        respx.get("https://broken.example.com/x").mock(
            return_value=httpx.Response(500, text="boom")
        )
        async with httpx.AsyncClient() as client:
            result = await fetch_hackernews_top(client, limit=1)

    assert len(result) == 1
    assert result[0].get("image_url") is None


# ──────────────────────────────────────────────────────────────────────────
# ARG-177: favicon fallback in _fetch_article_body
# ──────────────────────────────────────────────────────────────────────────


async def test_article_body_falls_back_to_favicon(monkeypatch) -> None:
    """An article page with no og/twitter/body image yields a domain favicon URL."""
    from argos.crawler import static_fetcher

    html = "<html><head><title>t</title></head><body><p>text only</p></body></html>"
    url = "https://news.example.com/a"

    # Stub is_robots_allowed so _get_with_retry doesn't hit the network.
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.is_robots_allowed",
        AsyncMock(return_value=True),
    )
    # Stub _is_safe_url so DNS resolution doesn't run in CI.
    monkeypatch.setattr(
        "argos.crawler.static_fetcher._is_safe_url",
        AsyncMock(return_value=True),
    )

    with respx.mock:
        respx.get(url).mock(
            return_value=httpx.Response(
                200, text=html, headers={"content-type": "text/html"}
            )
        )
        async with httpx.AsyncClient() as client:
            _body, image_url = await static_fetcher._fetch_article_body(client, url)

    assert image_url == "https://news.example.com/favicon.ico"
