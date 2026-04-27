from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain import run_brain_pipeline
from argos.brain.graph_state import BrainState
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
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
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
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, dict):
            dynamic_items.append(result)
        elif isinstance(result, Exception):
            logger.warning("dynamic fetch failed for %s: %r", url, result)

    return await filter_duplicate_urls(session, [*static_items, *dynamic_items])


async def run_full_pipeline(
    session: AsyncSession,
    dynamic_urls: list[str] | None = None,
) -> list[BrainState]:
    crawl_items = await run_full_crawl(session, dynamic_urls)
    results: list[BrainState] = []
    for item in crawl_items:
        source_url = item.get("source_url", "").strip()
        if not source_url:
            logger.warning(
                "run_full_pipeline: crawled item missing source_url, skipping: %r",
                item.get("title", "unknown"),
            )
            continue
        try:
            async with session.begin_nested():
                state = await run_brain_pipeline(
                    raw_text=item.get("raw_content") or "",
                    source_url=source_url,
                    session=session,
                )
            results.append(state)
        except Exception as exc:
            logger.warning(
                "run_full_pipeline: brain pipeline failed for %s: %r", source_url, exc
            )
    return results
