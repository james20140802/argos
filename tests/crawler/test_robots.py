from __future__ import annotations

import httpx
import pytest
import respx

from argos.crawler import _robots


@pytest.fixture(autouse=True)
def _reset_robots_cache():
    _robots._robots_cache.clear()
    _robots._robots_origin_locks.clear()
    yield
    _robots._robots_cache.clear()
    _robots._robots_origin_locks.clear()


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.asyncio
async def test_robots_auth_blocked_status_disallows(status):
    """401/403 robots.txt must be treated as disallow-all per RFC 9309."""
    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(status, text="")
        )
        allowed = await _robots.is_robots_allowed("https://example.com/anything")
    assert allowed is False


@pytest.mark.parametrize("status", [404, 410, 500])
@pytest.mark.asyncio
async def test_robots_other_non_2xx_allows_by_default(status):
    """Non-auth failures (404/5xx) keep RFC 9309 allow-by-default behavior."""
    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(status, text="")
        )
        allowed = await _robots.is_robots_allowed("https://example.com/anything")
    assert allowed is True


@pytest.mark.asyncio
async def test_robots_transport_error_fails_closed(monkeypatch):
    """Network failure on robots.txt must disallow, not silently allow."""

    class _BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, _url):
            raise httpx.ConnectError("DNS down")

    monkeypatch.setattr(_robots.httpx, "AsyncClient", _BoomClient)

    allowed = await _robots.is_robots_allowed("https://example.com/anything")
    assert allowed is False


@pytest.mark.asyncio
async def test_robots_transient_transport_error_is_not_cached(monkeypatch):
    """A one-off transport error must disallow that request but must NOT
    permanently poison the cache — recovery on the next call should allow."""
    call_log: list[str] = []

    class _FlakyClient:
        _calls = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            _FlakyClient._calls += 1
            call_log.append(url)
            if _FlakyClient._calls == 1:
                raise httpx.ConnectError("transient DNS blip")
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")

    monkeypatch.setattr(_robots.httpx, "AsyncClient", _FlakyClient)

    first = await _robots.is_robots_allowed("https://example.com/page")
    assert first is False  # fail-closed on transport error

    second = await _robots.is_robots_allowed("https://example.com/page")
    assert second is True  # cache was not poisoned — re-fetch succeeded
    assert len(call_log) == 2


@pytest.mark.asyncio
async def test_robots_follows_redirect_before_evaluating(monkeypatch):
    """A 301/302 redirect on /robots.txt must be followed so the terminal
    response — not the redirect — drives the crawl policy."""
    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(
                301, headers={"Location": "https://example.com/static/robots.txt"}
            )
        )
        respx.get("https://example.com/static/robots.txt").mock(
            return_value=httpx.Response(
                200, text="User-agent: *\nDisallow: /private/\n"
            )
        )
        allowed_private = await _robots.is_robots_allowed(
            "https://example.com/private/secret"
        )
    assert allowed_private is False

    _robots._robots_cache.clear()
    _robots._robots_origin_locks.clear()

    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(
                301, headers={"Location": "https://example.com/static/robots.txt"}
            )
        )
        respx.get("https://example.com/static/robots.txt").mock(
            return_value=httpx.Response(
                200, text="User-agent: *\nDisallow: /private/\n"
            )
        )
        allowed_public = await _robots.is_robots_allowed(
            "https://example.com/public/page"
        )
    assert allowed_public is True


@pytest.mark.asyncio
async def test_robots_cache_expires_after_ttl(monkeypatch):
    """Cached robots parsers must expire so a long-running crawler picks up
    new Disallow rules instead of reusing a stale parser indefinitely."""
    fake_now = [1000.0]

    def _fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr(_robots.time, "monotonic", _fake_monotonic)

    fetches: list[str] = []

    class _ChangingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            fetches.append(url)
            if len(fetches) == 1:
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            return httpx.Response(
                200, text="User-agent: *\nDisallow: /private/\n"
            )

    monkeypatch.setattr(_robots.httpx, "AsyncClient", _ChangingClient)

    first = await _robots.is_robots_allowed("https://example.com/private/page")
    assert first is True
    assert len(fetches) == 1

    fake_now[0] += 60.0
    cached = await _robots.is_robots_allowed("https://example.com/private/page")
    assert cached is True
    assert len(fetches) == 1

    fake_now[0] += _robots._ROBOTS_CACHE_TTL_SECONDS
    refreshed = await _robots.is_robots_allowed(
        "https://example.com/private/page"
    )
    assert refreshed is False
    assert len(fetches) == 2


@pytest.mark.asyncio
async def test_robots_can_fetch_exception_fails_closed(monkeypatch):
    """If parser.can_fetch raises (e.g. UnicodeEncodeError on malformed paths),
    enforcement must fail closed rather than silently allowing the request."""
    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
        )

        original_can_fetch = (
            _robots.urllib.robotparser.RobotFileParser.can_fetch
        )

        def _boom(self, useragent, url):
            raise UnicodeEncodeError("ascii", url, 0, 1, "boom")

        monkeypatch.setattr(
            _robots.urllib.robotparser.RobotFileParser, "can_fetch", _boom
        )

        try:
            allowed = await _robots.is_robots_allowed("https://example.com/anything")
        finally:
            monkeypatch.setattr(
                _robots.urllib.robotparser.RobotFileParser,
                "can_fetch",
                original_can_fetch,
            )

    assert allowed is False
