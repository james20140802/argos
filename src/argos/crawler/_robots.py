from __future__ import annotations

import asyncio
import time
import urllib.robotparser
from urllib.parse import urlsplit

import httpx

_ROBOTS_USER_AGENT = "argos-crawler"
_ROBOTS_FETCH_TIMEOUT = 10.0
_ROBOTS_MAX_REDIRECTS = 5
_ROBOTS_CACHE_TTL_SECONDS = 86400.0

# Hosts the operator has explicitly published as public read-only APIs for
# third-party use. Their generic robots.txt would block crawling, but the
# vendor (Y Combinator for HN) treats the endpoint as public. Keep this list
# tiny and only add hosts with an equivalent public-API contract.
_ROBOTS_ALLOWLISTED_HOSTS: frozenset[str] = frozenset(
    {
        "hacker-news.firebaseio.com",
    }
)

_robots_cache: dict[str, tuple[urllib.robotparser.RobotFileParser, float]] = {}
_robots_origin_locks: dict[str, asyncio.Lock] = {}
_robots_lock = asyncio.Lock()


class RobotsDisallowed(Exception):
    def __init__(self, url: str) -> None:
        super().__init__(f"robots.txt disallows: {url}")
        self.url = url


async def _fetch_robots_parser(
    origin: str,
    event_hooks: dict | None = None,
) -> tuple[urllib.robotparser.RobotFileParser, bool]:
    """Return (parser, cacheable). Transient transport failures disallow-all
    but are marked non-cacheable so a single blip does not permanently block
    the origin — the next request re-attempts the fetch."""
    parser = urllib.robotparser.RobotFileParser()
    robots_url = f"{origin}/robots.txt"
    try:
        async with httpx.AsyncClient(
            timeout=_ROBOTS_FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=_ROBOTS_MAX_REDIRECTS,
            event_hooks=event_hooks,
        ) as client:
            response = await client.get(robots_url)
    except httpx.HTTPError:
        parser.disallow_all = True
        return parser, False

    if 200 <= response.status_code < 300:
        parser.parse(response.text.splitlines())
    elif response.status_code in (401, 403):
        parser.disallow_all = True
    else:
        parser.parse([])
    return parser, True


def _live_cached_parser(
    origin: str,
) -> urllib.robotparser.RobotFileParser | None:
    """Return the cached parser only if its TTL has not expired."""
    cached = _robots_cache.get(origin)
    if cached is None:
        return None
    parser, cached_at = cached
    if time.monotonic() - cached_at >= _ROBOTS_CACHE_TTL_SECONDS:
        return None
    return parser


async def is_robots_allowed(
    url: str,
    user_agent: str = _ROBOTS_USER_AGENT,
    *,
    event_hooks: dict | None = None,
) -> bool:
    """``event_hooks``는 robots.txt를 새로 받을 때의 클라이언트에 붙는다(리다이렉트 포함).
    요청 간격을 지켜야 하는 호출자(Archive 폴백)가 robots 요청도 같은 throttle에 태운다."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if not parts.scheme or not parts.netloc:
        return False
    if parts.netloc.lower() in _ROBOTS_ALLOWLISTED_HOSTS:
        return True
    origin = f"{parts.scheme}://{parts.netloc}"

    parser = _live_cached_parser(origin)
    if parser is None:
        async with _robots_lock:
            origin_lock = _robots_origin_locks.get(origin)
            if origin_lock is None:
                origin_lock = asyncio.Lock()
                _robots_origin_locks[origin] = origin_lock
        async with origin_lock:
            parser = _live_cached_parser(origin)
            if parser is None:
                parser, cacheable = await _fetch_robots_parser(origin, event_hooks)
                if cacheable:
                    _robots_cache[origin] = (parser, time.monotonic())

    try:
        return parser.can_fetch(user_agent, url)
    except Exception:
        return False
