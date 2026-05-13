"""arXiv Atom feed fetcher for cs.AI / cs.LG / cs.CL papers (ARG-53).

Queries the arXiv public export API for papers submitted in the last N hours
and converts each Atom entry into the canonical item dict shape used by the
rest of the crawler pipeline.

Public API
----------
- ``fetch_arxiv_recent(*, hours, max_results, client) -> list[dict]``

Design notes
------------
- Uses a single GET to ``http://export.arxiv.org/api/query`` (arXiv's
  vendor-published public API — no robots gate required, mirrors the HN
  allowlist precedent in ``_robots._ROBOTS_ALLOWLISTED_HOSTS``).
- arXiv recommends a 3-second delay between requests for bulk/repeated use.
  We issue at most ONE GET per pipeline run so this threshold is not crossed.
  If paging is added in the future, honour the 3-second inter-request delay.
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
import logging
import time
import re

import feedparser
import httpx

from argos.crawler.user_agents import random_user_agent
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 8 * 1024  # 8 KB truncation limit — mirrors rss_fetcher

# arXiv Atom export API base URL (vendor-published public API contract)
_ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# Category query covering AI / machine learning / computational linguistics
_ARXIV_QUERY = "cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL"

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
    ``_source_category``.  ``source_url`` is normalised via
    ``_normalize_abs_url`` so deduplication survives versioned re-fetches.
    """
    title: str = getattr(entry, "title", "") or ""
    entry_id: str = getattr(entry, "id", "") or ""
    summary: str = getattr(entry, "summary", "") or ""

    source_url = _normalize_abs_url(entry_id) if entry_id else ""
    raw_content = _truncate(f"{title}\n\n{summary}")

    return {
        "title": title,
        "source_url": source_url,
        "raw_content": raw_content,
        "_source_category": CategoryType.ALPHA,
    }


async def fetch_arxiv_recent(
    *,
    hours: int = 24,
    max_results: int = 100,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Fetch arXiv papers in cs.AI/cs.LG/cs.CL submitted in the last *hours*.

    Parameters
    ----------
    hours:
        How far back to look (default: 24 h).
    max_results:
        Maximum number of entries to request from the API (default: 100).
    client:
        Optional pre-constructed ``httpx.AsyncClient`` (useful for testing);
        if ``None`` a fresh client is created and closed inside this call.

    Returns an empty list on any error (HTTP failure, parse failure, etc.) —
    the pipeline must not crash when arXiv is unreachable.
    """
    ua = random_user_agent()
    params = {
        "search_query": _ARXIV_QUERY,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }

    # Build query string manually to avoid double-encoding the '+OR+' operators
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_ARXIV_API_BASE}?{query_string}"

    cutoff_epoch = time.time() - hours * 3600

    try:
        _close_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=30, headers={"User-Agent": ua})

        try:
            response = await client.get(url, headers={"User-Agent": ua})
        finally:
            if _close_client:
                await client.aclose()

        if response.status_code >= 400:
            logger.warning(
                "arxiv_fetcher: HTTP %d from arXiv API — returning []",
                response.status_code,
            )
            return []

        content = response.content  # bytes

    except Exception as exc:
        logger.warning("arxiv_fetcher: request failed: %r — returning []", exc)
        return []

    try:
        parsed = await asyncio.to_thread(feedparser.parse, content)
    except Exception as exc:
        logger.warning("arxiv_fetcher: feedparser failed: %r — returning []", exc)
        return []

    items: list[dict] = []
    for entry in getattr(parsed, "entries", []):
        published_parsed = getattr(entry, "published_parsed", None)
        if published_parsed is not None:
            pub_epoch = calendar.timegm(published_parsed)  # UTC-correct
            if pub_epoch < cutoff_epoch:
                continue  # older than the cutoff window — skip

        item = _entry_to_dict(entry)
        if item["source_url"]:
            items.append(item)

    return items
