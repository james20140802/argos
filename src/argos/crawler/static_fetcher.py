from __future__ import annotations

import asyncio
import datetime as dt

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.crawler._robots import RobotsDisallowed, is_robots_allowed
from argos.crawler.dynamic_fetcher import _is_safe_url, extract_main_content
from argos.crawler.user_agents import random_user_agent
from argos.models.crawl_queue import CrawlQueue
from argos.models.tech_item import TechItem

_HN_CONCURRENCY = 8
_RETRY_MAX_ATTEMPTS = 3
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RAW_CONTENT_MAX_BYTES = 8 * 1024
_README_CANDIDATE_PATHS = ("README.md", "README.rst")


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_attempts: int = _RETRY_MAX_ATTEMPTS,
) -> httpx.Response:
    ua = random_user_agent()
    if not await is_robots_allowed(url, ua):
        raise RobotsDisallowed(url)
    for attempt in range(max_attempts):
        try:
            response = await client.get(
                url,
                headers={"User-Agent": ua},
            )
        except httpx.HTTPError:
            if attempt + 1 >= max_attempts:
                raise
            await asyncio.sleep(2**attempt)
            continue
        if (
            response.status_code in _RETRYABLE_STATUS_CODES
            and attempt + 1 < max_attempts
        ):
            await asyncio.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response
    raise httpx.HTTPError(f"exhausted retries for {url}")


def _truncate_raw_content(text: str, limit: int = _RAW_CONTENT_MAX_BYTES) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def _parse_github_repo_slug(href: str) -> tuple[str, str] | None:
    parts = [p for p in href.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


async def _fetch_github_readme(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
) -> str | None:
    for path in _README_CANDIDATE_PATHS:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        try:
            response = await _get_with_retry(client, url)
        except (httpx.HTTPError, RobotsDisallowed):
            continue
        text = response.text.strip()
        if text:
            return text
    return None


async def fetch_github_trending(
    client: httpx.AsyncClient,
    language: str | None = None,
) -> list[dict]:
    url = f"https://github.com/trending/{language}" if language else "https://github.com/trending"
    response = await _get_with_retry(client, url)

    soup = BeautifulSoup(response.text, "html.parser")
    parsed: list[tuple[str, str, str, tuple[str, str] | None]] = []

    for article in soup.select("article.Box-row"):
        anchor = article.select_one("h2.h3.lh-condensed a")
        if not anchor:
            continue
        href = (anchor.get("href") or "").strip()
        if not href.startswith("/"):
            continue

        source_url = f"https://github.com{href}"
        title = anchor.get_text(strip=True)
        description_tag = article.select_one("p.col-9.color-fg-muted") or article.find("p")
        description = description_tag.get_text(strip=True) if description_tag else title
        parsed.append((title, source_url, description, _parse_github_repo_slug(href)))

    semaphore = asyncio.Semaphore(_HN_CONCURRENCY)

    async def _readme_for(slug: tuple[str, str] | None) -> str | None:
        if slug is None:
            return None
        async with semaphore:
            return await _fetch_github_readme(client, slug[0], slug[1])

    async def _created_at_for(slug: tuple[str, str] | None) -> dt.datetime | None:
        if slug is None:
            return None
        async with semaphore:
            return await _fetch_github_repo_created_at(client, slug[0], slug[1])

    readmes, created_ats = await asyncio.gather(
        asyncio.gather(*(_readme_for(slug) for *_, slug in parsed)),
        asyncio.gather(*(_created_at_for(slug) for *_, slug in parsed)),
    )

    items: list[dict] = []
    for (title, source_url, description, _slug), readme, created_at in zip(
        parsed, readmes, created_ats
    ):
        if readme:
            combined = f"{description}\n\n{readme}" if description else readme
        else:
            combined = description
        items.append(
            {
                "title": title,
                "source_url": source_url,
                "raw_content": _truncate_raw_content(combined),
                "_published_at": created_at,
            }
        )

    return items


async def _fetch_github_repo_created_at(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
) -> dt.datetime | None:
    """Call the GitHub REST API to get a repo's created_at timestamp.

    Returns a UTC-aware datetime or None on any error (4xx, 5xx, parse error).
    Does NOT raise — callers must tolerate None gracefully.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        response = await _get_with_retry(client, url)
    except (httpx.HTTPError, RobotsDisallowed):
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    created_str = data.get("created_at")
    if not isinstance(created_str, str) or not created_str:
        return None
    try:
        return dt.datetime.fromisoformat(
            created_str.replace("Z", "+00:00")
        ).astimezone(dt.timezone.utc)
    except (ValueError, TypeError):
        return None


async def _fetch_article_body(client: httpx.AsyncClient, url: str) -> str:
    if not await _is_safe_url(url):
        return ""
    try:
        response = await _get_with_retry(client, url)
    except (httpx.HTTPError, RobotsDisallowed):
        return ""
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type:
        return ""
    _title, body = extract_main_content(response.text)
    return body.strip()


async def fetch_hackernews_top(
    client: httpx.AsyncClient,
    limit: int = 30,
) -> list[dict]:
    response = await _get_with_retry(
        client,
        "https://hacker-news.firebaseio.com/v0/topstories.json",
    )
    try:
        payload = response.json()
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    top_ids: list[int] = [i for i in payload[:limit] if isinstance(i, int)]

    semaphore = asyncio.Semaphore(_HN_CONCURRENCY)

    async def _fetch_item(item_id: int) -> dict | None:
        async with semaphore:
            try:
                r = await _get_with_retry(
                    client,
                    f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json",
                )
            except httpx.HTTPError:
                return None
            try:
                data = r.json()
            except ValueError:
                return None
            if not isinstance(data, dict):
                return None
            fallback_url = f"https://news.ycombinator.com/item?id={item_id}"
            candidate = data.get("url")
            is_external = False
            if isinstance(candidate, str) and candidate.lower().startswith(
                ("http://", "https://")
            ):
                url = candidate
                is_external = True
            else:
                url = fallback_url
            title = data.get("title") or ""
            text = data.get("text") or ""
            if not isinstance(title, str):
                title = ""
            if not isinstance(text, str):
                text = ""
            if text:
                raw_content = f"{title} {text}".strip()
            elif is_external:
                body = await _fetch_article_body(client, url)
                raw_content = f"{title}\n\n{body}".strip() if body else title
            else:
                raw_content = title
            time_val = data.get("time")
            published_at: dt.datetime | None = None
            if isinstance(time_val, (int, float)):
                try:
                    published_at = dt.datetime.fromtimestamp(time_val, tz=dt.timezone.utc)
                except (ValueError, OSError):
                    pass
            return {
                "title": title,
                "source_url": url,
                "raw_content": _truncate_raw_content(raw_content),
                "_published_at": published_at,
            }

    results = await asyncio.gather(*(_fetch_item(i) for i in top_ids))
    return [item for item in results if item is not None]


async def filter_duplicate_urls(
    session: AsyncSession,
    items: list[dict],
) -> list[dict]:
    if not items:
        return []

    candidate_urls = [item["source_url"] for item in items]

    result_tech = await session.execute(
        select(TechItem.source_url).where(TechItem.source_url.in_(candidate_urls))
    )
    result_queue = await session.execute(
        select(CrawlQueue.source_url).where(CrawlQueue.source_url.in_(candidate_urls))
    )
    existing_urls: set[str] = set(result_tech.scalars().all()) | set(result_queue.scalars().all())

    deduped: list[dict] = []
    seen: set[str] = set(existing_urls)
    for item in items:
        url = item["source_url"]
        if url in seen:
            continue
        seen.add(url)
        deduped.append(item)
    return deduped
