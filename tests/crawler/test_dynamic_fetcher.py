from __future__ import annotations

import asyncio
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

    async def _fake_resolve(host: str):
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


def _make_playwright_mock(page_html: str, page_url: str = "https://example.com/article"):
    """Build a nested AsyncMock chain mimicking async_playwright context manager."""
    mock_route = MagicMock()
    mock_route.request.resource_type = "image"

    mock_page = AsyncMock()
    mock_page.content.return_value = page_html
    mock_page.url = page_url
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
        # RFC 6598 CGNAT range (100.64.0.0/10): neither private nor reserved in
        # Python's ipaddress module, but still non-global — must be blocked as SSRF.
        "http://100.64.0.1/",
        "http://100.127.255.254/",
    ],
)
@pytest.mark.asyncio
async def test_is_safe_url_blocks_unsafe_targets(url):
    assert await _is_safe_url(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/path",
        "http://news.ycombinator.com/item?id=1",
    ],
)
@pytest.mark.asyncio
async def test_is_safe_url_allows_public_targets(url, _public_dns):
    assert await _is_safe_url(url) is True


@pytest.mark.asyncio
async def test_is_safe_url_allows_public_literal_ip():
    assert await _is_safe_url("https://8.8.8.8/") is True


@pytest.mark.parametrize(
    "url",
    [
        "http://foo.localhost/",
        "http://service.local/",
        "http://metadata.internal/",
    ],
)
@pytest.mark.asyncio
async def test_is_safe_url_blocks_special_use_suffixes(url):
    assert await _is_safe_url(url) is False


@pytest.mark.asyncio
async def test_is_safe_url_blocks_dns_rebinding(monkeypatch):
    """Hostnames that resolve to private IPs must be rejected."""
    import ipaddress as _ipaddress

    async def _rebind(host: str):
        return [_ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._resolve_hostname",
        _rebind,
    )
    assert await _is_safe_url("http://attacker.example.com/") is False


@pytest.mark.asyncio
async def test_is_safe_url_blocks_unresolvable_host(monkeypatch):
    async def _empty(host: str):
        return []

    monkeypatch.setattr(
        "argos.crawler.dynamic_fetcher._resolve_hostname",
        _empty,
    )
    assert await _is_safe_url("http://nonexistent.example.invalid/") is False


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1/broken",  # malformed IPv6 bracket syntax
        "http://[bad:addr/",
    ],
)
@pytest.mark.asyncio
async def test_is_safe_url_rejects_malformed_urls(url):
    assert await _is_safe_url(url) is False


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

@pytest.mark.asyncio
async def test_resolve_hostname_handles_unicode_error(monkeypatch):
    """Malformed IDNA hostnames must not leak UnicodeError."""
    from argos.crawler.dynamic_fetcher import _resolve_hostname

    async def _raise_unicode(*args, **kwargs):
        raise UnicodeError("IDNA label too long")

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", _raise_unicode)
    assert await _resolve_hostname("a" * 1000 + ".example.com") == []


@pytest.mark.asyncio
async def test_is_safe_url_handles_idna_unicode_error(monkeypatch):
    """_is_safe_url must fail closed when resolution raises UnicodeError."""

    async def _raise_unicode(*args, **kwargs):
        raise UnicodeError("IDNA label too long")

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", _raise_unicode)
    assert await _is_safe_url("http://" + "x" * 512 + ".example.com/") is False


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

    async def _fake_resolve(host: str):
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


# ---------------------------------------------------------------------------
# Test 6: SSRF redirect bypass — final URL re-validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_handler_aborts_unsafe_intermediate_redirect(_public_dns):
    """Intermediate redirect hops must be aborted by the route handler,
    even if the terminal URL is safe."""
    from argos.crawler import dynamic_fetcher as df

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, mock_page = _make_playwright_mock(sample_html)

    captured = {"handler": None}

    async def _capture_route(_pattern, handler):
        captured["handler"] = handler

    mock_page.route.side_effect = _capture_route

    aborts: list[str] = []
    continues: list[str] = []

    async def _goto(*_args, **_kwargs):
        # Simulate Playwright invoking the route handler for a redirect to
        # a link-local SSRF target (AWS metadata service).
        unsafe_request = MagicMock()
        unsafe_request.resource_type = "document"
        unsafe_request.url = "http://169.254.169.254/latest/meta-data/"
        unsafe_request.is_navigation_request = MagicMock(return_value=True)

        unsafe_route = MagicMock()
        unsafe_route.request = unsafe_request

        async def _abort():
            aborts.append(unsafe_request.url)

        async def _continue():
            continues.append(unsafe_request.url)

        unsafe_route.abort = _abort
        unsafe_route.continue_ = _continue

        await captured["handler"](unsafe_route)

    mock_page.goto.side_effect = _goto

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        await df.fetch_dynamic_page("https://example.com/article")

    assert aborts == ["http://169.254.169.254/latest/meta-data/"]
    assert continues == []


@pytest.mark.asyncio
async def test_route_handler_aborts_unsafe_non_navigation_request(_public_dns):
    """Non-navigation subresources (xhr/fetch/script) targeting SSRF hosts
    must be aborted too — not only navigation requests."""
    from argos.crawler import dynamic_fetcher as df

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, mock_page = _make_playwright_mock(sample_html)

    captured = {"handler": None}

    async def _capture_route(_pattern, handler):
        captured["handler"] = handler

    mock_page.route.side_effect = _capture_route

    aborts: list[str] = []
    continues: list[str] = []

    async def _goto(*_args, **_kwargs):
        # A cloud metadata endpoint fetched via xhr, NOT navigation.
        xhr_request = MagicMock()
        xhr_request.resource_type = "xhr"
        xhr_request.url = "http://169.254.169.254/latest/meta-data/"
        xhr_request.is_navigation_request = MagicMock(return_value=False)

        xhr_route = MagicMock()
        xhr_route.request = xhr_request

        async def _abort():
            aborts.append(xhr_request.url)

        async def _continue():
            continues.append(xhr_request.url)

        xhr_route.abort = _abort
        xhr_route.continue_ = _continue

        await captured["handler"](xhr_route)

    mock_page.goto.side_effect = _goto

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        await df.fetch_dynamic_page("https://example.com/article")

    assert aborts == ["http://169.254.169.254/latest/meta-data/"]
    assert continues == []


@pytest.mark.asyncio
async def test_route_handler_continues_safe_navigation(_public_dns):
    """Safe navigation requests must pass through the route handler unblocked."""
    from argos.crawler import dynamic_fetcher as df

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, mock_page = _make_playwright_mock(sample_html)

    captured = {"handler": None}

    async def _capture_route(_pattern, handler):
        captured["handler"] = handler

    mock_page.route.side_effect = _capture_route

    aborts: list[str] = []
    continues: list[str] = []

    async def _goto(*_args, **_kwargs):
        safe_request = MagicMock()
        safe_request.resource_type = "document"
        safe_request.url = "https://example.com/article"
        safe_request.is_navigation_request = MagicMock(return_value=True)

        safe_route = MagicMock()
        safe_route.request = safe_request

        async def _abort():
            aborts.append(safe_request.url)

        async def _continue():
            continues.append(safe_request.url)

        safe_route.abort = _abort
        safe_route.continue_ = _continue

        await captured["handler"](safe_route)

    mock_page.goto.side_effect = _goto

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        await df.fetch_dynamic_page("https://example.com/article")

    assert continues == ["https://example.com/article"]
    assert aborts == []


@pytest.mark.asyncio
async def test_fetch_dynamic_page_uses_redirected_final_url_as_source(_public_dns):
    """When a wrapper URL redirects to a canonical URL, source_url must be the
    final URL so dedup collapses wrapper duplicates against static sources."""
    from argos.crawler import dynamic_fetcher as df

    wrapper_url = "https://t.co/abcd"
    canonical_url = "https://example.com/article"

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, _ = _make_playwright_mock(sample_html, page_url=canonical_url)

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        result = await df.fetch_dynamic_page(wrapper_url)

    assert result is not None
    assert result["source_url"] == canonical_url


@pytest.mark.asyncio
async def test_fetch_dynamic_page_blocks_unsafe_redirect(_public_dns):
    """If page.goto redirects to an unsafe host, fetch must return None."""
    from argos.crawler import dynamic_fetcher as df

    original_url = "https://example.com/article"
    # Simulate a redirect to an internal IP (SSRF target)
    redirected_url = "http://169.254.169.254/latest/meta-data/"

    sample_html = (FIXTURES_DIR / "sample_article.html").read_text()
    mock_pw_cm, mock_page = _make_playwright_mock(sample_html, page_url=redirected_url)

    post_redirect_checked = {"fired": False}
    original_is_safe_url = df._is_safe_url

    async def _tracking_is_safe_url(url: str) -> bool:
        if url == redirected_url:
            post_redirect_checked["fired"] = True
        return await original_is_safe_url(url)

    with patch("argos.crawler.dynamic_fetcher.async_playwright", return_value=mock_pw_cm):
        with patch("argos.crawler.dynamic_fetcher._is_safe_url", side_effect=_tracking_is_safe_url):
            result = await df.fetch_dynamic_page(original_url)

    assert result is None
    assert post_redirect_checked["fired"], "Post-redirect _is_safe_url check was not called"
