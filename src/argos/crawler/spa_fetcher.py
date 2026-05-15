from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urljoin, urlsplit

from argos.config import SPASourceConfig
from argos.crawler._robots import is_robots_allowed
from argos.crawler.dynamic_fetcher import _load_page_html, fetch_dynamic_page
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)

_ARTICLE_CONCURRENCY = 3


async def fetch_spa_source(config: SPASourceConfig) -> list[dict]:
    """Load a SPA listing page, extract article links, and crawl each one."""
    try:
        allowed = await is_robots_allowed(config.listing_url)
    except Exception as exc:
        logger.warning("spa_fetcher: robots check failed for %s: %r", config.listing_url, exc)
        return []

    if not allowed:
        logger.info("spa_fetcher: robots.txt disallows %s — skipping", config.listing_url)
        return []

    try:
        html, _ = await _load_page_html(config.listing_url, timeout_ms=20_000)
    except Exception as exc:
        logger.warning("spa_fetcher: failed to load listing page %s: %r", config.listing_url, exc)
        return []

    pattern = re.compile(config.link_pattern)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)

    seen: set[str] = set()
    article_urls: list[str] = []
    for href in hrefs:
        if not pattern.search(href):
            continue
        abs_url = urljoin(config.base_url, href) if not href.startswith("http") else href
        expected_netloc = urlsplit(config.base_url).netloc
        if urlsplit(abs_url).netloc != expected_netloc:
            continue
        abs_url = abs_url.split("#")[0]
        parts = urlsplit(abs_url)
        if not parts.scheme or not parts.netloc:
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        article_urls.append(abs_url)
        if len(article_urls) >= config.max_items:
            break

    if not article_urls:
        logger.info(
            "spa_fetcher: no links matched pattern %r on %s",
            config.link_pattern,
            config.listing_url,
        )
        return []

    semaphore = asyncio.Semaphore(_ARTICLE_CONCURRENCY)

    async def _fetch_one(url: str) -> dict | None:
        async with semaphore:
            return await fetch_dynamic_page(url)

    gathered = await asyncio.gather(
        *(_fetch_one(url) for url in article_urls),
        return_exceptions=True,
    )

    source_tag = f"spa:{config.name}" if config.name else f"spa:{urlsplit(config.listing_url).netloc}"
    category = CategoryType(config.category)

    items: list[dict] = []
    for url, result in zip(article_urls, gathered):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            logger.warning("spa_fetcher: failed to fetch %s: %r", url, result)
            continue
        if result is None:
            logger.info("spa_fetcher: fetch_dynamic_page returned None for %s", url)
            continue
        result["_source"] = source_tag
        result["_source_category"] = category
        result["_published_at"] = None
        items.append(result)

    return items


async def run_spa_fetchers(sources: list[SPASourceConfig]) -> list[dict]:
    """Fetch all SPA sources concurrently; per-source failures log and skip."""
    if not sources:
        return []

    results = await asyncio.gather(
        *(fetch_spa_source(s) for s in sources),
        return_exceptions=True,
    )

    combined: list[dict] = []
    for source, result in zip(sources, results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            label = source.name or source.listing_url
            logger.warning("spa_fetcher: gather error for %s: %r", label, result)
            continue
        combined.extend(result)

    return combined
