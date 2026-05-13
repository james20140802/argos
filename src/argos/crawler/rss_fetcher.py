"""RSS feed fetcher for AI company blogs and community feeds (ARG-52).

Wraps the synchronous ``feedparser`` library in ``asyncio.to_thread`` so
callers stay fully async.  Each source is fetched concurrently via
``asyncio.gather``; per-source failures are logged and swallowed so a single
broken feed does not abort the entire batch.

Public API
----------
- ``fetch_rss_feed(url, category) -> list[dict]``
- ``run_rss_fetchers(feeds) -> list[dict]``
"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

import feedparser

from argos.config import RSSFeedConfig
from argos.crawler._robots import is_robots_allowed
from argos.crawler.user_agents import random_user_agent
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 8 * 1024  # 8 KB truncation limit


def _truncate(text: str, max_bytes: int = _MAX_CONTENT_BYTES) -> str:
    """Truncate *text* so its UTF-8 encoding does not exceed *max_bytes*."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _entry_to_dict(entry: object, category: CategoryType) -> dict | None:
    """Convert a feedparser entry to the canonical item dict.

    Returns ``None`` when the entry is missing a required field (``title`` or
    ``link``).
    """
    title: str | None = getattr(entry, "title", None)
    link: str | None = getattr(entry, "link", None)

    if not title or not link:
        return None

    # Prefer the full content over summary when available
    content_parts = getattr(entry, "content", None)
    if content_parts:
        body = content_parts[0].get("value", "") if isinstance(content_parts, list) else ""
    else:
        body = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""

    raw_content = _truncate(f"{title}\n\n{body}")

    return {
        "title": title,
        "source_url": link,
        "raw_content": raw_content,
        "_source_category": category,
    }


async def fetch_rss_feed(url: str, category: CategoryType) -> list[dict]:
    """Fetch a single RSS/Atom feed and return a list of item dicts.

    Robots gate is checked first; if disallowed the feed is skipped and ``[]``
    is returned.  feedparser is called in a thread so the event loop stays
    unblocked.  Any exception is caught, logged, and results in an empty list.
    """
    ua = random_user_agent()
    try:
        allowed = await is_robots_allowed(url, ua)
    except Exception as exc:
        logger.warning("rss_fetcher: robots check failed for %s: %r — skipping", url, exc)
        return []

    if not allowed:
        logger.info("rss_fetcher: robots.txt disallows %s — skipping", url)
        return []

    try:
        parsed = await asyncio.to_thread(feedparser.parse, url, agent=ua)
    except Exception as exc:
        logger.warning("rss_fetcher: feedparser failed for %s: %r", url, exc)
        return []

    # feedparser signals HTTP-level errors via bozo / status attributes
    status = getattr(parsed, "status", 200)
    if status and status >= 400:
        logger.warning("rss_fetcher: HTTP %d for %s — skipping", status, url)
        return []

    items: list[dict] = []
    for entry in getattr(parsed, "entries", []):
        item = _entry_to_dict(entry, category)
        if item is not None:
            items.append(item)

    return items


async def run_rss_fetchers(feeds: list[RSSFeedConfig]) -> list[dict]:
    """Fetch all configured RSS feeds concurrently.

    ``category`` strings from config (``"Mainstream"`` / ``"Alpha"``) are
    converted to ``CategoryType`` enum values before being stamped onto items.
    """
    if not feeds:
        return []

    async def _safe_fetch(feed: RSSFeedConfig) -> list[dict]:
        category = CategoryType(feed.category)
        try:
            return await fetch_rss_feed(feed.url, category)
        except Exception as exc:  # pragma: no cover — belt-and-suspenders
            host = urlsplit(feed.url).netloc or feed.url
            logger.warning("rss_fetcher: unexpected error for %s: %r", host, exc)
            return []

    results = await asyncio.gather(*(_safe_fetch(f) for f in feeds), return_exceptions=True)

    combined: list[dict] = []
    for feed, result in zip(feeds, results, strict=True):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            host = urlsplit(feed.url).netloc or feed.url
            logger.warning("rss_fetcher: gather error for %s: %r", host, result)
            continue
        combined.extend(result)

    return combined
