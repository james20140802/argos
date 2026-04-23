from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.crawler.dynamic_fetcher import (
    BLOCKED_RESOURCE_TYPES,
    extract_main_content,
    fetch_dynamic_page,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_playwright_mock(page_html: str):
    """Build a nested AsyncMock chain mimicking async_playwright context manager."""
    mock_route = MagicMock()
    mock_route.request.resource_type = "image"

    mock_page = AsyncMock()
    mock_page.content.return_value = page_html
    mock_page.route = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.close = AsyncMock()

    mock_context = AsyncMock()
    mock_context.new_page.return_value = mock_page
    mock_context.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context.return_value = mock_context
    mock_browser.close = AsyncMock()

    mock_chromium = MagicMock()
    mock_chromium.launch = AsyncMock(return_value=mock_browser)

    mock_pw = MagicMock()
    mock_pw.chromium = mock_chromium

    mock_pw_cm = AsyncMock()
    mock_pw_cm.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_pw_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_pw_cm, mock_page


# ---------------------------------------------------------------------------
# Test 1: constant sanity check
# ---------------------------------------------------------------------------

def test_blocked_resource_types_contains_image():
    assert "image" in BLOCKED_RESOURCE_TYPES
    assert "stylesheet" in BLOCKED_RESOURCE_TYPES
    assert "font" in BLOCKED_RESOURCE_TYPES
    assert "media" in BLOCKED_RESOURCE_TYPES


# ---------------------------------------------------------------------------
# Test 2: extract_main_content strips scripts and keeps article body
# ---------------------------------------------------------------------------

def test_extract_main_content_returns_title_and_body():
    html = (
        "<html><head><title>Title</title></head><body>"
        "<h1>Title</h1><article><p>Body</p></article>"
        "<script>ads</script></body></html>"
    )
    title, raw_content = extract_main_content(html)
    assert title == "Title"
    assert "Body" in raw_content
    assert "ads" not in raw_content


# ---------------------------------------------------------------------------
# Test 3: successful fetch returns proper dict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_dynamic_page_returns_dict_on_success():
    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, _ = _make_playwright_mock(sample_html)

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        result = await fetch_dynamic_page("https://example.com/article")

    assert result is not None
    assert "title" in result
    assert result["source_url"] == "https://example.com/article"
    assert "raw_content" in result


# ---------------------------------------------------------------------------
# Test 4: retries on TimeoutError, succeeds on third attempt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_dynamic_page_retries_on_timeout():
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, mock_page = _make_playwright_mock(sample_html)

    call_count = 0

    async def goto_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise PlaywrightTimeoutError("timeout")

    mock_page.goto.side_effect = goto_side_effect

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_dynamic_page("https://example.com/article", max_retries=3)

    assert result is not None
    assert call_count == 3


# ---------------------------------------------------------------------------
# Test 5: returns None after exhausting max_retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_dynamic_page_returns_none_after_max_retries():
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, mock_page = _make_playwright_mock(sample_html)

    call_count = 0

    async def always_timeout(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise PlaywrightTimeoutError("timeout")

    mock_page.goto.side_effect = always_timeout

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await fetch_dynamic_page("https://example.com/article", max_retries=3)

    assert result is None
    assert call_count == 4  # initial attempt + 3 retries
