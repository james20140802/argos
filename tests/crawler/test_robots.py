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
