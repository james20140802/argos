from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

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
            result["_source"] = "dynamic"
            dynamic_items.append(result)
        elif isinstance(result, Exception):
            logger.warning("dynamic fetch failed for %s: %r", url, result)

    return await filter_duplicate_urls(session, [*static_items, *dynamic_items])


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
