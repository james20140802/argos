from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.crawler.dynamic_fetcher import (
    BLOCKED_RESOURCE_TYPES,
    _is_safe_url,
    extract_main_content,
    fetch_dynamic_page,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def _public_dns(monkeypatch):
    """Force hostname resolution to a fixed public IP so SSRF checks stay hermetic."""

    def _fake_resolve(host: str):
        import ipaddress as _ipaddress

        return [_ipaddress.ip_address("93.184.216.34")]

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._resolve_hostname",
        _fake_resolve,
    )

    async def _always_allow(_url: str) -> bool:
        return True

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._is_robots_allowed",
        _always_allow,
    )


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

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost:8080/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "http:///no-host",
    ],
)
def test_is_safe_url_blocks_unsafe_targets(url):
    assert _is_safe_url(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/path",
        "http://news.ycombinator.com/item?id=1",
    ],
)
def test_is_safe_url_allows_public_targets(url, _public_dns):
    assert _is_safe_url(url) is True


def test_is_safe_url_allows_public_literal_ip():
    assert _is_safe_url("https://8.8.8.8/") is True


@pytest.mark.parametrize(
    "url",
    [
        "http://foo.localhost/",
        "http://service.local/",
        "http://metadata.internal/",
    ],
)
def test_is_safe_url_blocks_special_use_suffixes(url):
    assert _is_safe_url(url) is False


def test_is_safe_url_blocks_dns_rebinding(monkeypatch):
    """Hostnames that resolve to private IPs must be rejected."""
    import ipaddress as _ipaddress

    def _rebind(host: str):
        return [_ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._resolve_hostname",
        _rebind,
    )
    assert _is_safe_url("http://attacker.example.com/") is False


def test_is_safe_url_blocks_unresolvable_host(monkeypatch):
    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._resolve_hostname",
        lambda host: [],
    )
    assert _is_safe_url("http://nonexistent.example.invalid/") is False


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1/broken",  # malformed IPv6 bracket syntax
        "http://[bad:addr/",
    ],
)
def test_is_safe_url_rejects_malformed_urls(url):
    assert _is_safe_url(url) is False


def test_extract_main_content_handles_empty_html():
    title, body = extract_main_content("")
    assert title == ""
    assert body == ""


def test_extract_main_content_handles_garbage_html():
    title, body = extract_main_content("<<not really html<<<")
    assert isinstance(title, str)
    assert isinstance(body, str)


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
async def test_fetch_dynamic_page_returns_dict_on_success(_public_dns):
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
async def test_fetch_dynamic_page_retries_on_timeout(_public_dns):
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

def test_resolve_hostname_handles_unicode_error(monkeypatch):
    """Malformed IDNA hostnames must not leak UnicodeError."""
    from argos.crawler.dynamic_fetcher import _resolve_hostname

    def _raise_unicode(*args, **kwargs):
        raise UnicodeError("IDNA label too long")

    monkeypatch.setattr("socket.getaddrinfo", _raise_unicode)
    assert _resolve_hostname("a" * 1000 + ".example.com") == []


def test_is_safe_url_handles_idna_unicode_error(monkeypatch):
    """_is_safe_url must fail closed when resolution raises UnicodeError."""

    def _raise_unicode(*args, **kwargs):
        raise UnicodeError("IDNA label too long")

    monkeypatch.setattr("socket.getaddrinfo", _raise_unicode)
    assert _is_safe_url("http://" + "x" * 512 + ".example.com/") is False


@pytest.mark.asyncio
async def test_is_robots_allowed_blocks_disallowed_path(monkeypatch):
    import httpx
    import respx

    from argos.crawler import dynamic_fetcher as df

    df._robots_cache.clear()
    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(
                200, text="User-agent: *\nDisallow: /private/\n"
            )
        )
        allowed = await df._is_robots_allowed("https://example.com/private/secret")
    assert allowed is False
    df._robots_cache.clear()


@pytest.mark.asyncio
async def test_is_robots_allowed_permits_allowed_path():
    import httpx
    import respx

    from argos.crawler import dynamic_fetcher as df

    df._robots_cache.clear()
    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(
                200, text="User-agent: *\nDisallow: /private/\n"
            )
        )
        allowed = await df._is_robots_allowed("https://example.com/public/page")
    assert allowed is True
    df._robots_cache.clear()


@pytest.mark.asyncio
async def test_fetch_dynamic_page_skips_when_robots_disallows(monkeypatch):
    """fetch_dynamic_page must return None without loading when robots.txt blocks."""
    from argos.crawler import dynamic_fetcher as df

    def _fake_resolve(host: str):
        import ipaddress as _ipaddress

        return [_ipaddress.ip_address("93.184.216.34")]

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._resolve_hostname", _fake_resolve
    )

    async def _deny(_url: str) -> bool:
        return False

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._is_robots_allowed", _deny
    )

    called = {"goto": 0}

    def _fail_if_called(*args, **kwargs):
        called["goto"] += 1
        raise AssertionError("playwright should not run when robots disallows")

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher.async_playwright", _fail_if_called
    )

    result = await df.fetch_dynamic_page("https://example.com/page")
    assert result is None
    assert called["goto"] == 0


@pytest.mark.asyncio
async def test_fetch_dynamic_page_returns_none_after_max_retries(_public_dns):
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
