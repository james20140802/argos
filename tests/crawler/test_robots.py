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
