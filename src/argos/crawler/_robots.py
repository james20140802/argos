from __future__ import annotations

import asyncio
import urllib.robotparser
from urllib.parse import urlsplit

import httpx

_ROBOTS_USER_AGENT = "argos-crawler"
_ROBOTS_FETCH_TIMEOUT = 10.0

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
_robots_origin_locks: dict[str, asyncio.Lock] = {}
_robots_lock = asyncio.Lock()


class RobotsDisallowed(Exception):
    def __init__(self, url: str) -> None:
        super().__init__(f"robots.txt disallows: {url}")
        self.url = url


async def _fetch_robots_parser(origin: str) -> urllib.robotparser.RobotFileParser:
    parser = urllib.robotparser.RobotFileParser()
    robots_url = f"{origin}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=_ROBOTS_FETCH_TIMEOUT) as client:
            response = await client.get(robots_url)
    except httpx.HTTPError:
        parser.disallow_all = True
        return parser

    if 200 <= response.status_code < 300:
        parser.parse(response.text.splitlines())
    else:
        parser.parse([])
    return parser


async def is_robots_allowed(url: str, user_agent: str = _ROBOTS_USER_AGENT) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if not parts.scheme or not parts.netloc:
        return False
    origin = f"{parts.scheme}://{parts.netloc}"

    parser = _robots_cache.get(origin)
    if parser is None:
        async with _robots_lock:
            origin_lock = _robots_origin_locks.get(origin)
            if origin_lock is None:
                origin_lock = asyncio.Lock()
                _robots_origin_locks[origin] = origin_lock
        async with origin_lock:
            parser = _robots_cache.get(origin)
            if parser is None:
                parser = await _fetch_robots_parser(origin)
                _robots_cache[origin] = parser

    try:
        return parser.can_fetch(user_agent, url)
    except Exception:
        return True
