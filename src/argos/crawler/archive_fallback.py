"""Internet Archive Wayback 보관본으로 막힌 사이트의 지난 기사 본문을 채운다 (ARG-309).

범위는 의도적으로 좁다.

* **사람이 실행하는 ``argos refetch-bodies``에서만** 쓴다. 매일 도는
  ``argos run`` / 큐 경로(``crawler.pipeline``)에서는 절대 부르지 않는다.
* :data:`ARCHIVE_FALLBACK_HOSTS`에 있는 호스트(지금은 ``openai.com`` 하나)의
  기사 URL에만 쓴다. openai.com 원 사이트는 Cloudflare 봇 챌린지로 막혀 있고
  우리는 그것을 푸는 시도를 하지 않는다 — 이 모듈은 남이 보관한 사본을 받을 뿐이다.
* Internet Archive는 기부로 운영되는 비영리라, 이 모듈이 내는 모든 요청
  (가용성 API + 스냅샷, 재시도·리다이렉트 포함)을 합쳐 초당 1건 이하로 보낸다.
* 문서화된 공개 API(archive.org/help/wayback_api.php)의 ``available``과
  ``id_`` 원본 스냅샷만 쓴다. robots.txt는 일반 가져오기와 같은 경로
  (``_get_with_retry``)에서 지킨다. ``_robots.py``의 allowlist는 건드리지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Awaitable, Callable
from urllib.parse import urlencode, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from argos.crawler.body_enrich import host_of
from argos.crawler.dynamic_fetcher import extract_main_content
from argos.crawler.static_fetcher import _get_with_retry, _truncate_raw_content

__all__ = [
    "ARCHIVE_FALLBACK_HOSTS",
    "ArchiveFetcher",
    "RequestThrottle",
    "extract_article_text",
    "is_archive_eligible",
    "snapshot_url",
]

logger = logging.getLogger(__name__)

#: Wayback 보충을 허용하는 호스트. 넓히려면 사용자 결정이 필요하다.
ARCHIVE_FALLBACK_HOSTS = frozenset({"openai.com"})

_AVAILABILITY_ENDPOINT = "https://archive.org/wayback/available"
_ARCHIVE_HOSTS = frozenset({"archive.org", "web.archive.org"})
_MIN_INTERVAL_SECONDS = 1.0
_MAX_HOPS = 5
_TIMEOUT_SECONDS = 30.0
_MAX_CONSECUTIVE_LOOKUP_ERRORS = 3
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_TIMESTAMP = re.compile(r"^\d{8,14}$")
# readability가 엉뚱한 블록을 고른 것으로 볼 기준 (실측: 1,645자 vs <article> 10,955자).
_ARTICLE_FALLBACK_RATIO = 1.5
_WHITESPACE = re.compile(r"\s+")


def is_archive_eligible(url: str) -> bool:
    """허용 호스트의 기사 URL(홈페이지 제외)인가."""
    try:
        path = urlsplit(url).path
    except ValueError:
        return False
    return host_of(url) in ARCHIVE_FALLBACK_HOSTS and path.strip("/") != ""


def snapshot_url(timestamp: str, original_url: str) -> str:
    """``id_`` 플래그는 Wayback 툴바·링크 재작성이 없는 원본 HTML을 준다."""
    return f"https://web.archive.org/web/{timestamp}id_/{original_url}"


def extract_article_text(html: str) -> tuple[str, str]:
    """readability 결과를 쓰되, ``<article>``/``<main>`` 텍스트가 훨씬 길면 그것을 쓴다."""
    title, body = extract_main_content(html)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "aside", "footer"]):
        tag.decompose()
    node = soup.find("article") or soup.find("main")
    if node is not None:
        alt = _WHITESPACE.sub(" ", node.get_text(" ")).strip()
        if len(alt) > _ARTICLE_FALLBACK_RATIO * len(body):
            body = alt
    return title, body


class RequestThrottle:
    """요청 사이 최소 간격을 지킨다. 한 클라이언트의 모든 요청이 이걸 거친다."""

    def __init__(
        self,
        min_interval: float = _MIN_INTERVAL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            if self._last is not None:
                delay = self._last + self.min_interval - self._clock()
                if delay > 0:
                    await self._sleep(delay)
            self._last = self._clock()


class ArchiveFetcher:
    """Wayback에서 기사 본문을 가져온다. ``async with``로 쓴다."""

    def __init__(self, *, throttle: RequestThrottle | None = None) -> None:
        self.throttle = throttle or RequestThrottle()
        self.fetched = 0  # 보관본에서 본문을 얻은 건수
        self.disabled = False  # 가용성 API가 연달아 실패하면 이번 실행에서는 접는다
        self._lookup_errors = 0
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ArchiveFetcher":
        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS,
            follow_redirects=False,
            event_hooks={"request": [self._before_request]},
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _before_request(self, request: httpx.Request) -> None:
        await self.throttle.wait()

    async def _get(self, url: str) -> httpx.Response | None:
        """Archive 호스트 안에서만 리다이렉트를 따라가며 GET. 실패하면 None."""
        assert self._client is not None, "use ArchiveFetcher as an async context manager"
        current = url
        for _ in range(_MAX_HOPS + 1):
            try:
                # 재시도 없음: Archive가 느려졌을 때 우리가 요청을 더 쌓지 않는다.
                response = await _get_with_retry(self._client, current, max_attempts=1)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    # 속도 제한에 걸렸다. 이번 실행에서는 더 두드리지 않는다.
                    logger.warning("Internet Archive rate-limited us (429); skipping archive for this run")
                    self.disabled = True
                    return None
                # httpx 0.28의 raise_for_status는 3xx에도 예외를 던진다.
                if exc.response.status_code not in _REDIRECT_STATUS:
                    logger.info("archive GET %s -> %s", current, exc.response.status_code)
                    return None
                response = exc.response
            except Exception as exc:  # noqa: BLE001 - robots 거부·전송 오류 모두 "없음"
                logger.info("archive GET failed for %s: %r", current, exc)
                return None

            if response.status_code not in _REDIRECT_STATUS:
                return response
            location = response.headers.get("location")
            if not location:
                return None
            nxt = urljoin(current, location)
            if (urlsplit(nxt).hostname or "").lower() not in _ARCHIVE_HOSTS:
                logger.info("archive redirect leaves archive hosts: %s", nxt)
                return None
            current = nxt
        return None

    async def lookup(self, url: str) -> str | None:
        """가장 가까운 200 스냅샷의 타임스탬프. 없으면 None."""
        response = await self._get(f"{_AVAILABILITY_ENDPOINT}?{urlencode({'url': url})}")
        if response is None:
            self._lookup_errors += 1
            if self._lookup_errors >= _MAX_CONSECUTIVE_LOOKUP_ERRORS:
                logger.warning("Wayback availability API keeps failing; skipping archive for this run")
                self.disabled = True
            return None
        self._lookup_errors = 0
        try:
            closest = response.json()["archived_snapshots"]["closest"]
        except (ValueError, KeyError, TypeError):
            return None
        if not isinstance(closest, dict) or not closest.get("available"):
            return None
        if str(closest.get("status", "200")) != "200":
            return None
        timestamp = str(closest.get("timestamp", ""))
        return timestamp if _TIMESTAMP.match(timestamp) else None

    async def fetch(self, url: str) -> dict | None:
        """``{"title", "raw_content", "source_url", "archived_at"}`` 또는 None."""
        if self.disabled or not is_archive_eligible(url):
            return None
        timestamp = await self.lookup(url)
        if timestamp is None:
            return None
        response = await self._get(snapshot_url(timestamp, url))
        if response is None:
            return None
        title, body = extract_article_text(response.text)
        body = body.strip()
        if not body:
            return None
        self.fetched += 1
        return {
            "title": title or "",
            "raw_content": _truncate_raw_content(body),
            "source_url": url,
            "archived_at": timestamp,
        }
