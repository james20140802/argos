from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from argos.crawler.dynamic_fetcher import fetch_dynamic_page
from argos.crawler.static_fetcher import (
    fetch_github_trending,
    fetch_hackernews_top,
    filter_duplicate_urls,
)

logger = logging.getLogger(__name__)

_DYNAMIC_CONCURRENCY = 3


async def run_static_pipeline(session: AsyncSession) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        results = await asyncio.gather(
            fetch_github_trending(client),
            fetch_hackernews_top(client),
            return_exceptions=True,
        )

    combined: list[dict] = []
    for source, result in zip(("github_trending", "hackernews"), results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("static source %s failed: %r", source, result)
            continue
        combined.extend(result)

    return await filter_duplicate_urls(session, combined)


async def run_full_crawl(
    session: AsyncSession,
    dynamic_urls: list[str] | None = None,
) -> list[dict]:
    static_items = await run_static_pipeline(session)

    if not dynamic_urls:
        return static_items

    semaphore = asyncio.Semaphore(_DYNAMIC_CONCURRENCY)

    async def _bounded_fetch(url: str) -> dict | None:
        async with semaphore:
            return await fetch_dynamic_page(url)

    dynamic_results = await asyncio.gather(
        *(_bounded_fetch(url) for url in dynamic_urls),
        return_exceptions=True,
    )
    dynamic_items: list[dict] = []
    for url, result in zip(dynamic_urls, dynamic_results, strict=True):
        if isinstance(result, dict):
            dynamic_items.append(result)
        elif isinstance(result, BaseException):
            logger.warning("dynamic fetch failed for %s: %r", url, result)

    return [*static_items, *dynamic_items]
