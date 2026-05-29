"""arXiv Atom feed fetcher for cs.AI / cs.LG / cs.CL papers (ARG-53).

Queries the arXiv public export API for papers submitted in the last N hours
and converts each Atom entry into the canonical item dict shape used by the
rest of the crawler pipeline.

Public API
----------
- ``fetch_arxiv_recent(*, hours, max_results, client) -> list[dict]``

Design notes
------------
- Pages through ``http://export.arxiv.org/api/query`` (arXiv's vendor-published
  public API — no robots gate required, mirrors the HN allowlist precedent in
  ``_robots._ROBOTS_ALLOWLISTED_HOSTS``) using the ``start=`` offset parameter
  until entries fall outside the lookback window or the feed is exhausted.
- arXiv recommends a 3-second delay between requests for bulk/repeated use.
  A 3-second ``asyncio.sleep`` is inserted between page fetches (not before
  the first request).  A safety cap of ``_MAX_PAGES`` pages prevents runaway
  loops on a misbehaving feed.
- ``feedparser`` is used to parse the Atom XML bytes (already a dep via
  ARG-52).  We pass raw bytes from httpx rather than a URL so the HTTP call
  stays under httpx (consistent User-Agent rotation + timeout).
- ``raw_content`` is built from ``title + abstract`` only — no full-text, no
  author names.  LaTeX markup in summaries is left intact; triage handles it.
- ``source_url`` is normalised to the canonical
  ``https://arxiv.org/abs/{bare_id}`` form (no ``vN`` suffix) so
  ``filter_duplicate_urls`` deduplicates re-fetches of updated paper versions.
"""
from __future__ import annotations

import asyncio
import calendar
import datetime as dt
import logging
import time
import re

import feedparser
import httpx

from argos.crawler._html_utils import clean_title
from argos.crawler.user_agents import random_user_agent
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 8 * 1024  # 8 KB truncation limit — mirrors rss_fetcher

# arXiv Atom export API base URL (vendor-published public API contract)
_ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# Category query covering AI / machine learning / computational linguistics
_ARXIV_QUERY = "cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"

# Maximum number of pages to fetch in a single call (safety cap against
# misbehaving feeds that never return entries older than the cutoff).
_MAX_PAGES = 20

# Delay between paginated requests, as recommended by arXiv for bulk access.
_INTER_REQUEST_DELAY = 3.0  # seconds

# Regex that strips a trailing version suffix from a bare arXiv paper ID.
# Handles both new-style IDs (e.g. "2401.12345v2") and legacy IDs that
# contain slashes (e.g. "hep-ex/0307015v1").
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def _truncate(text: str, max_bytes: int = _MAX_CONTENT_BYTES) -> str:
    """Truncate *text* so its UTF-8 encoding does not exceed *max_bytes*."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _normalize_abs_url(entry_id: str) -> str:
    """Return the canonical ``https://arxiv.org/abs/{bare_id}`` URL.

    Strips any ``vN`` version suffix and normalises the scheme to https.
    Handles both new-style IDs (e.g. ``2401.12345``) and legacy IDs with
    archive prefixes (e.g. ``hep-ex/0307015``).

    >>> _normalize_abs_url("http://arxiv.org/abs/2401.12345v2")
    'https://arxiv.org/abs/2401.12345'
    >>> _normalize_abs_url("https://arxiv.org/abs/2401.12345")
    'https://arxiv.org/abs/2401.12345'
    >>> _normalize_abs_url("http://arxiv.org/abs/hep-ex/0307015v1")
    'https://arxiv.org/abs/hep-ex/0307015'
    """
    idx = entry_id.find("/abs/")
    if idx != -1:
        bare = _VERSION_SUFFIX_RE.sub("", entry_id[idx + 5:])
        return f"https://arxiv.org/abs/{bare}"
    # Fallback: force https and strip trailing vN suffix
    url = entry_id.replace("http://", "https://", 1)
    url = _VERSION_SUFFIX_RE.sub("", url)
    return url


def _entry_to_dict(entry: object) -> dict:
    """Convert a feedparser Atom entry to the canonical crawler item dict.

    Returns a dict with keys: ``title``, ``source_url``, ``raw_content``,
    ``_source_category``, ``_published_at``.  ``source_url`` is normalised via
    ``_normalize_abs_url`` so deduplication survives versioned re-fetches.
    """
    title: str = getattr(entry, "title", "") or ""
    entry_id: str = getattr(entry, "id", "") or ""
    summary: str = getattr(entry, "summary", "") or ""
    title = clean_title(title)

    source_url = _normalize_abs_url(entry_id) if entry_id else ""
    raw_content = _truncate(f"{title}\n\n{summary}")

    published_parsed = getattr(entry, "published_parsed", None)
    published_at: dt.datetime | None = None
    if published_parsed is not None:
        try:
            published_at = dt.datetime(*published_parsed[:6], tzinfo=dt.timezone.utc)
        except (TypeError, ValueError):
            pass

    return {
        "title": title,
        "source_url": source_url,
        "raw_content": raw_content,
        "_source_category": CategoryType.ALPHA,
        "_published_at": published_at,
    }


async def fetch_arxiv_recent(
    *,
    hours: int = 24,
    max_results: int = 100,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Fetch arXiv papers in cs.AI/cs.LG/cs.CL submitted in the last *hours*.

    Pages through the arXiv API using the ``start=`` offset parameter until
    entries fall outside the lookback window or the feed is exhausted.  A
    3-second delay is observed between page fetches per arXiv's bulk-access
    recommendation.  At most ``_MAX_PAGES`` pages are fetched as a safety cap.

    Parameters
    ----------
    hours:
        How far back to look (default: 24 h).
    max_results:
        Number of entries to request per page (default: 100).
    client:
        Optional pre-constructed ``httpx.AsyncClient`` (useful for testing);
        if ``None`` a fresh client is created and closed inside this call.

    Returns an empty list on any error (HTTP failure, parse failure, etc.) —
    the pipeline must not crash when arXiv is unreachable.
    """
    ua = random_user_agent()
    cutoff_epoch = time.time() - hours * 3600

    _close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=30, headers={"User-Agent": ua})

    items: list[dict] = []
    start = 0

    try:
        for page in range(_MAX_PAGES):
            if page > 0:
                await asyncio.sleep(_INTER_REQUEST_DELAY)

            params = {
                "search_query": _ARXIV_QUERY,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": str(max_results),
                "start": str(start),
            }
            # Build query string manually to avoid double-encoding the '+OR+' operators
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{_ARXIV_API_BASE}?{query_string}"

            try:
                response = await client.get(url, headers={"User-Agent": ua})
            except Exception as exc:
                logger.warning(
                    "arxiv_fetcher: request failed on page %d: %r — stopping pagination",
                    page,
                    exc,
                )
                break

            if response.status_code >= 400:
                logger.warning(
                    "arxiv_fetcher: HTTP %d from arXiv API on page %d — stopping pagination",
                    response.status_code,
                    page,
                )
                break

            try:
                parsed = await asyncio.to_thread(feedparser.parse, response.content)
            except Exception as exc:
                logger.warning(
                    "arxiv_fetcher: feedparser failed on page %d: %r — stopping pagination",
                    page,
                    exc,
                )
                break

            entries = getattr(parsed, "entries", [])
            if not entries:
                # Feed exhausted
                break

            reached_cutoff = False
            for entry in entries:
                published_parsed = getattr(entry, "published_parsed", None)
                if published_parsed is not None:
                    pub_epoch = calendar.timegm(published_parsed)  # UTC-correct
                    if pub_epoch < cutoff_epoch:
                        # Results are sorted descending; everything after this
                        # point is even older — stop pagination.
                        reached_cutoff = True
                        break

                item = _entry_to_dict(entry)
                if item["source_url"]:
                    items.append(item)

            if reached_cutoff:
                break

            if len(entries) < max_results:
                # Fewer entries than requested — feed is exhausted
                break

            start += max_results

    finally:
        if _close_client:
            await client.aclose()

    return items
