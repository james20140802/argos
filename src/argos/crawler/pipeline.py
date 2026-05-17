from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain import run_batch_brain_pipeline
from argos.brain.graph_state import BrainState
from argos.brain.preflight import is_preflight_reject
from argos.config import settings
from argos.crawler.arxiv_fetcher import fetch_arxiv_recent
from argos.crawler.dynamic_fetcher import fetch_dynamic_page
from argos.crawler.rss_fetcher import run_rss_fetchers
from argos.crawler.spa_fetcher import run_spa_fetchers
from argos.crawler.static_fetcher import (
    fetch_github_trending,
    fetch_hackernews_top,
    filter_duplicate_urls,
)
from argos.models.crawl_queue import CrawlQueue
from argos.models.tech_item import CategoryType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from argos.progress import ProgressReporter

logger = logging.getLogger(__name__)

_DYNAMIC_CONCURRENCY = 3


@dataclass
class PipelineSummary:
    crawled_total: int
    per_source: dict[str, int] = field(default_factory=dict)
    queue_selected: int = 0
    triage_pass: int = 0
    saved_new: int = 0
    genealogy_skipped: int = 0
    trust_skipped: int = 0
    preflight_filtered: int = 0
    duration_seconds: float = 0.0
    queue_remaining: int = 0


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
    feeds = settings.user.rss.feeds
    items = await run_rss_fetchers(feeds)
    for item in items:
        host = urlsplit(item.get("source_url", "")).netloc or "rss"
        item["_source"] = f"rss:{host}"
    return await filter_duplicate_urls(session, items)


async def run_arxiv_pipeline(session: AsyncSession) -> list[dict]:
    """Fetch recent arXiv cs.AI/cs.LG/cs.CL papers and return deduplicated item dicts."""
    items = await fetch_arxiv_recent()
    for item in items:
        item["_source"] = "arxiv"
    return await filter_duplicate_urls(session, items)


async def run_spa_pipeline(session: AsyncSession) -> list[dict]:
    """Fetch all configured SPA sources and return deduplicated item dicts."""
    sources = settings.user.spa.sources
    items = await run_spa_fetchers(sources)
    return await filter_duplicate_urls(session, items)


async def run_full_crawl(
    session: AsyncSession,
    dynamic_urls: list[str] | None = None,
) -> list[dict]:
    # Run sub-pipelines sequentially: AsyncSession is not safe for concurrent
    # use, so overlapping DB calls (filter_duplicate_urls) in both branches
    # would cause session-state errors and silently drop items.
    static_items: list[dict] = []
    try:
        static_items = await run_static_pipeline(session)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("run_full_crawl: static pipeline failed: %r", exc)

    rss_items: list[dict] = []
    try:
        rss_items = await run_rss_pipeline(session)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("run_full_crawl: rss pipeline failed: %r", exc)

    arxiv_items: list[dict] = []
    try:
        arxiv_items = await run_arxiv_pipeline(session)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("run_full_crawl: arxiv pipeline failed: %r", exc)

    spa_items: list[dict] = []
    try:
        spa_items = await run_spa_pipeline(session)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("run_full_crawl: spa pipeline failed: %r", exc)

    if not dynamic_urls:
        return await filter_duplicate_urls(session, [*static_items, *rss_items, *arxiv_items, *spa_items])

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

    return await filter_duplicate_urls(session, [*static_items, *rss_items, *arxiv_items, *spa_items, *dynamic_items])


async def _upsert_crawl_queue(session: AsyncSession, items: list[dict]) -> int:
    """Insert new items into crawl_queue; skip on source_url conflict.

    Returns the number of rows submitted (pre-conflict-resolution count).
    """
    if not items:
        return 0
    rows = [
        {
            "source_url": item["source_url"],
            "raw_content": item.get("raw_content"),
            "source": item.get("_source"),
            "source_category": (
                item["_source_category"].value
                if isinstance(item.get("_source_category"), CategoryType)
                else item.get("_source_category")
            ),
            "published_at": item.get("_published_at"),
        }
        for item in items
    ]
    stmt = pg_insert(CrawlQueue).values(rows).on_conflict_do_nothing(index_elements=["source_url"])
    await session.execute(stmt)
    return len(rows)


async def _pop_from_queue(session: AsyncSession, limit: int) -> list[CrawlQueue]:
    """Select up to *limit* items from crawl_queue, newest published_at first.

    When *limit* is 0, all items are returned (unlimited mode).
    NULL published_at rows sort after all non-NULL rows; queued_at ASC breaks ties.
    """
    stmt = select(CrawlQueue).order_by(
        CrawlQueue.published_at.desc().nulls_last(),
        CrawlQueue.queued_at.asc(),
    )
    if limit > 0:
        stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _delete_from_queue(session: AsyncSession, urls: list[str]) -> None:
    if not urls:
        return
    await session.execute(
        delete(CrawlQueue).where(CrawlQueue.source_url.in_(urls))
    )


async def _queue_count(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(CrawlQueue))
    return result.scalar_one()


async def run_full_pipeline(
    session: AsyncSession,
    dynamic_urls: list[str] | None = None,
    *,
    progress: "ProgressReporter | None" = None,
) -> tuple[list[BrainState], PipelineSummary]:
    """Drive crawl → preflight → brain → save end to end.

    Parameters
    ----------
    session:
        Active async SQLAlchemy session.
    dynamic_urls:
        Optional list of ad-hoc URLs (``argos run --url ...``) fed through
        the dynamic Playwright fetcher in addition to the static / RSS /
        arXiv / SPA sources.
    progress:
        Optional :class:`argos.progress.ProgressReporter`. When provided,
        each pipeline stage drives ``start_stage`` / ``advance`` /
        ``finish_stage`` calls so the CLI can render a live progress bar.
        ``None`` (default) keeps the previous behavior (no progress
        reporting at all) so callers that don't need a UI are unaffected.
    """

    def _ps(name: str, total: int | None = None) -> None:
        if progress is not None:
            progress.start_stage(name, total=total)

    def _pa(name: str) -> None:
        if progress is not None:
            progress.advance(name)

    def _pf(name: str) -> None:
        if progress is not None:
            progress.finish_stage(name)

    def _pt(name: str, total: int) -> None:
        if progress is not None:
            progress.update_total(name, total)

    start = time.monotonic()
    daily_limit = settings.user.run.daily_limit

    # ── Stage 1: crawl all sources, dedup against tech_items + crawl_queue ──
    _ps("crawl")
    crawl_items = await run_full_crawl(session, dynamic_urls)
    _pt("crawl", len(crawl_items))
    for _ in crawl_items:
        _pa("crawl")
    _pf("crawl")

    per_source: dict[str, int] = {}
    for item in crawl_items:
        src = item.get("_source", "unknown")
        per_source[src] = per_source.get(src, 0) + 1

    # ── Stage 2: upsert new items into the queue ──────────────────────────
    queued_total = await _upsert_crawl_queue(session, crawl_items)
    await session.flush()

    # ── Stage 3: select today's batch from queue ──────────────────────────
    queue_rows = await _pop_from_queue(session, daily_limit)
    selected_items = [
        {
            "source_url": row.source_url,
            "raw_content": row.raw_content or "",
            "_source": row.source,
            "_source_category": (
                CategoryType(row.source_category) if row.source_category else None
            ),
        }
        for row in queue_rows
    ]

    # ── Stage 4: preflight filter ─────────────────────────────────────────
    preflight_filtered = 0
    if settings.user.triage.preflight_filter:
        _ps("preflight", total=len(selected_items))
        filtered: list[dict] = []
        for item in selected_items:
            if is_preflight_reject(item.get("raw_content") or ""):
                logger.info("preflight rejected: %s", item.get("source_url", "unknown"))
                preflight_filtered += 1
            else:
                filtered.append(item)
            _pa("preflight")
        selected_items = filtered
        _pf("preflight")

    valid_items = [item for item in selected_items if item.get("source_url", "").strip()]
    skipped_no_url = len(selected_items) - len(valid_items)
    if skipped_no_url:
        logger.warning(
            "run_full_pipeline: %d queued item(s) missing source_url, skipping",
            skipped_no_url,
        )

    # ── Stage 5: brain pipeline ───────────────────────────────────────────
    # The genealogy stage's total is only known after the trust-gate inside
    # run_batch_brain_pipeline runs, so we start it indeterminate. Embed total
    # equals the count of valid post-triage items, which we also can't know
    # up front — Rich handles None gracefully as a spinner-only bar.
    if progress is not None:
        _ps("triage", total=len(valid_items))
        _ps("embed")
        _ps("genealogy")
        _ps("save", total=len(valid_items))
        results: list[BrainState] = await run_batch_brain_pipeline(
            valid_items,
            session,
            on_triage_item_done=progress.callback_for("triage"),
            on_embed_item_done=progress.callback_for("embed"),
            on_genealogy_item_done=progress.callback_for("genealogy"),
            on_save_item_done=progress.callback_for("save"),
        )
        _pf("triage")
        _pf("embed")
        _pf("genealogy")
        _pf("save")
    else:
        results = await run_batch_brain_pipeline(valid_items, session)

    # ── Stage 6: remove processed rows from queue ─────────────────────────
    # Items where save_node raised (is_valid=True, saved=False) stay in the
    # queue so a transient DB error triggers a retry on the next run.
    # Triage-rejected items (is_valid=False) are deleted — they are not tech
    # items and retrying them wastes compute.
    save_failed_urls: set[str] = {
        s["source_url"]
        for s in results
        if s.get("is_valid") and not s.get("saved") and s.get("source_url")
    }
    urls_to_delete = [
        row.source_url
        for row in queue_rows
        if row.source_url not in save_failed_urls
    ]
    await _delete_from_queue(session, urls_to_delete)
    queue_remaining = await _queue_count(session)

    await session.commit()

    duration = time.monotonic() - start
    triage_pass = sum(1 for s in results if s.get("is_valid", False))
    saved_new = sum(1 for s in results if s.get("saved", False))
    genealogy_skipped = sum(
        1 for s in results
        if s.get("genealogy_skipped") and s.get("genealogy_skip_reason") != "low_trust"
    )
    trust_skipped = sum(
        1 for s in results
        if s.get("genealogy_skip_reason") == "low_trust"
    )

    summary = PipelineSummary(
        crawled_total=queued_total,
        per_source=per_source,
        queue_selected=len(queue_rows),
        triage_pass=triage_pass,
        saved_new=saved_new,
        genealogy_skipped=genealogy_skipped,
        trust_skipped=trust_skipped,
        preflight_filtered=preflight_filtered,
        duration_seconds=duration,
        queue_remaining=queue_remaining,
    )
    return results, summary
