"""기사 링크 기반 외부 증거 — ARG-313 자동 갈래 배정의 보조 신호.

**이름으로 검색하지 않는다.** 이름 검색은 동명이인 저장소(``Pi``, ``Modal``)를 끌고
와 증거를 오염시킨다. 대신 그 기술이 나온 **기사 본문에 걸린 링크**만 따라간다:

- ``github.com/{owner}/{repo}`` — 저장소 이름(영숫자만)이 기술 이름(영숫자만)을
  품을 때만. ``api.github.com/repos/{owner}/{repo}``의 ``topics`` -> ``gh:{topic}``.
- ``huggingface.co/{org}/{model}`` — 같은 규칙. ``/api/models/{id}``의
  ``pipeline_tag`` -> ``hf:{tag}``.

크롤러 규칙을 그대로 따른다: User-Agent 로테이션, robots.txt 확인, 429/5xx 지수
백오프. 결과는 파일 캐시(``~/.cache/argos/tech_external.json``, TTL)에 두고, 한
실행의 요청 수에 상한이 있다. 네트워크가 실패하면 **그 증거만** 비운다 — 배정 자체를
막지 않는다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import httpx

from argos.crawler._robots import is_robots_allowed
from argos.crawler.user_agents import random_user_agent

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path.home() / ".cache" / "argos" / "tech_external.json"
MIN_KEY_CHARS = 3
MAX_LINKS = 3
_RETRYABLE = {429, 500, 502, 503, 504}
_TIMEOUT = 15.0

_GH_LINK = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_HF_LINK = re.compile(r"huggingface\.co/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_GH_RESERVED = frozenset(
    {"orgs", "topics", "features", "about", "sponsors", "marketplace", "settings", "login",
     "search", "apps", "collections", "trending", "explore", "enterprise", "pricing", "site",
     "security", "customer-stories", "readme", "events", "github"}
)
_HF_RESERVED = frozenset(
    {"datasets", "spaces", "docs", "blog", "papers", "models", "api", "collections",
     "organizations", "settings", "join", "login", "learn", "tasks", "pricing", "enterprise",
     "chat", "posts", "changelog"}
)


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def links_for(texts: Iterable[str], name: str) -> tuple[list[str], list[str]]:
    """(github "owner/repo" 목록, huggingface "org/model" 목록). 많이 걸린 순, 최대 3개씩."""
    key = _alnum(name)
    if len(key) < MIN_KEY_CHARS:
        return [], []
    gh: Counter[str] = Counter()
    hf: Counter[str] = Counter()
    for text in texts:
        for owner, repo in _GH_LINK.findall(text or ""):
            repo = repo.removesuffix(".git").rstrip(".")
            if owner.lower() in _GH_RESERVED or key not in _alnum(repo):
                continue
            gh[f"{owner}/{repo}"] += 1
        for org, model in _HF_LINK.findall(text or ""):
            model = model.rstrip(".")
            if org.lower() in _HF_RESERVED or key not in _alnum(model):
                continue
            hf[f"{org}/{model}"] += 1
    rank = lambda c: [k for k, _n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))][:MAX_LINKS]  # noqa: E731
    return rank(gh), rank(hf)


RobotsFn = Callable[[str, str], Awaitable[bool]]


class ExternalEvidence:
    """링크 -> 태그. 한 배치 실행 동안 하나를 만들어 쓰고 끝에 ``save()``."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        cache_path: Path = DEFAULT_CACHE_PATH,
        ttl_seconds: float = 7 * 86400,
        max_requests: int = 40,
        robots: RobotsFn = is_robots_allowed,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        now: Callable[[], float] = time.time,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self.cache_path = Path(cache_path)
        self.ttl = ttl_seconds
        self.max_requests = max_requests
        self.requests = 0
        self._robots = robots
        self._sleep = sleep
        self._now = now
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._cache: dict[str, dict[str, Any]] = self._load_cache()
        self._dirty = False

    # ---------------------------------------------------------------- cache
    def _load_cache(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)
            self._dirty = False
        except OSError as exc:
            logger.warning("tech_external: cache save failed: %r", exc)

    def _cached(self, url: str) -> list[str] | None:
        entry = self._cache.get(url)
        if not isinstance(entry, dict):
            return None
        if self._now() - float(entry.get("at", 0)) >= self.ttl:
            return None
        return list(entry.get("tags") or [])

    def _store(self, url: str, tags: list[str]) -> None:
        self._cache[url] = {"tags": tags, "at": self._now()}
        self._dirty = True

    # ---------------------------------------------------------------- http
    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)
        return self._client

    async def _get_json(self, url: str) -> tuple[int, Any] | None:
        """(상태, JSON) 또는 None(요청 안 함·실패). 404는 (404, None)."""
        if self.requests >= self.max_requests:
            return None
        ua = random_user_agent()
        try:
            allowed = await self._robots(url, ua)
        except Exception:  # noqa: BLE001
            allowed = False
        if not allowed:
            logger.info("tech_external: robots.txt disallows %s", url)
            return None
        self.requests += 1
        for attempt in range(self._max_attempts):
            try:
                resp = await self._http().get(
                    url, headers={"User-Agent": ua, "Accept": "application/json"}
                )
            except httpx.HTTPError as exc:
                if attempt + 1 >= self._max_attempts:
                    logger.info("tech_external: %s failed: %r", url, exc)
                    return None
                await self._sleep(self._base_delay * 2**attempt)
                continue
            if resp.status_code in _RETRYABLE and attempt + 1 < self._max_attempts:
                await self._sleep(self._base_delay * 2**attempt)
                continue
            if resp.status_code == 404:
                return 404, None
            if resp.status_code != 200:
                return None
            try:
                return 200, resp.json()
            except ValueError:
                return None
        return None

    async def _tags(self, url: str, extract: Callable[[Any], list[str]]) -> list[str]:
        cached = self._cached(url)
        if cached is not None:
            return cached
        got = await self._get_json(url)
        if got is None:
            return []  # 네트워크 실패·상한·robots: 이번 실행에서 그 증거만 비운다
        status, data = got
        tags = extract(data) if status == 200 else []
        self._store(url, tags)
        return tags

    async def tags_for(self, gh_repos: Iterable[str], hf_models: Iterable[str]) -> list[str]:
        out: list[str] = []
        for slug in gh_repos:
            out += await self._tags(
                f"https://api.github.com/repos/{slug}",
                lambda d: [f"gh:{t}" for t in (d or {}).get("topics") or [] if isinstance(t, str)],
            )
        for mid in hf_models:
            out += await self._tags(
                f"https://huggingface.co/api/models/{mid}",
                lambda d: [f"hf:{d['pipeline_tag']}"]
                if isinstance(d, dict) and isinstance(d.get("pipeline_tag"), str)
                else [],
            )
        return list(dict.fromkeys(out))
