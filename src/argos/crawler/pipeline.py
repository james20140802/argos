from __future__ import annotations

import asyncio

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from argos.crawler.dynamic_fetcher import fetch_dynamic_page
from argos.crawler.static_fetcher import (
    fetch_github_trending,
    fetch_hackernews_top,
    filter_duplicate_urls,
)

_DYNAMIC_CONCURRENCY = 3


async def run_static_pipeline(session: AsyncSession) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        gh_items, hn_items = await asyncio.gather(
            fetch_github_trending(client),
            fetch_hackernews_top(client),
        )

    combined = [*gh_items, *hn_items]
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
    dynamic_items = [r for r in dynamic_results if isinstance(r, dict)]

    return [*static_items, *dynamic_items]
