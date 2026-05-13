from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain import run_brain_pipeline
from argos.brain.graph_state import BrainState
from argos.config import UserConfig
from argos.crawler.dynamic_fetcher import fetch_dynamic_page
from argos.crawler.rss_fetcher import run_rss_fetchers
from argos.crawler.static_fetcher import (
    fetch_github_trending,
    fetch_hackernews_top,
    filter_duplicate_urls,
)

logger = logging.getLogger(__name__)

_DYNAMIC_CONCURRENCY = 3


@dataclass
class PipelineSummary:
    crawled_total: int
    per_source: dict[str, int] = field(default_factory=dict)
    triage_pass: int = 0
    saved_new: int = 0
    genealogy_skipped: int = 0
    duration_seconds: float = 0.0


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
        for item in result:
            item["_source"] = source
        combined.extend(result)

    return await filter_duplicate_urls(session, combined)


async def run_rss_pipeline(session: AsyncSession) -> list[dict]:
    """Fetch all configured RSS feeds and return deduplicated item dicts."""
    feeds = UserConfig.load().rss.feeds
    items = await run_rss_fetchers(feeds)
    for item in items:
        host = urlsplit(item.get("source_url", "")).netloc or "rss"
        item["_source"] = f"rss:{host}"
    return await filter_duplicate_urls(session, items)


async def run_full_crawl(
    session: AsyncSession,
    dynamic_urls: list[str] | None = None,
) -> list[dict]:
    static_task = asyncio.create_task(run_static_pipeline(session))
    rss_task = asyncio.create_task(run_rss_pipeline(session))

    static_result, rss_result = await asyncio.gather(
        static_task, rss_task, return_exceptions=True
    )

    static_items: list[dict] = []
    if isinstance(static_result, asyncio.CancelledError):
        raise static_result
    elif isinstance(static_result, Exception):
        logger.warning("run_full_crawl: static pipeline failed: %r", static_result)
    else:
        static_items = static_result

    rss_items: list[dict] = []
    if isinstance(rss_result, asyncio.CancelledError):
        raise rss_result
    elif isinstance(rss_result, Exception):
        logger.warning("run_full_crawl: rss pipeline failed: %r", rss_result)
    else:
        rss_items = rss_result

    if not dynamic_urls:
        return await filter_duplicate_urls(session, [*static_items, *rss_items])

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
            result["_source"] = "dynamic"
            dynamic_items.append(result)
        elif isinstance(result, Exception):
            logger.warning("dynamic fetch failed for %s: %r", url, result)

    return await filter_duplicate_urls(session, [*static_items, *rss_items, *dynamic_items])


async def run_full_pipeline(
    session: AsyncSession,
    dynamic_urls: list[str] | None = None,
) -> tuple[list[BrainState], PipelineSummary]:
    start = time.monotonic()
    crawl_items = await run_full_crawl(session, dynamic_urls)

    # Compute per-source counts from crawled items
    per_source: dict[str, int] = {}
    for item in crawl_items:
        src = item.get("_source", "unknown")
        per_source[src] = per_source.get(src, 0) + 1

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
                # RSS (ARG-52) and arXiv (ARG-53) fetchers stamp a
                # "_source_category" key on their item dicts to hint triage.
                # GitHub/HN items carry no such key, so source_category
                # defaults to None — leaving the LLM to decide without a hint.
                source_category = item.get("_source_category")
                if source_category is not None:
                    state = await run_brain_pipeline(
                        raw_text=item.get("raw_content") or "",
                        source_url=source_url,
                        session=session,
                        source_category=source_category,
                    )
                else:
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
    await session.commit()

    duration = time.monotonic() - start
    triage_pass = sum(1 for s in results if s.get("is_valid", False))
    saved_new = sum(1 for s in results if s.get("saved", False))
    genealogy_skipped = sum(1 for s in results if s.get("genealogy_skipped", False))

    summary = PipelineSummary(
        crawled_total=len(crawl_items),
        per_source=per_source,
        triage_pass=triage_pass,
        saved_new=saved_new,
        genealogy_skipped=genealogy_skipped,
        duration_seconds=duration,
    )
    return results, summary
