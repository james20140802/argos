"""Shared service for `argos add <URL>` — manual URL pipeline injection (ARG-105).

Provides a single async entry point :func:`add_url` callable from both the
CLI (`argos add <URL>`) and the Slack slash command (`/argos add <URL>`).

Pipeline:
    1. Parse URL + scheme whitelist (http/https only).
    2. SSRF guard via :func:`crawler.dynamic_fetcher._is_safe_url`.
    3. robots.txt check via :func:`crawler._robots.is_robots_allowed`.
    4. Dedup against ``tech_items.source_url`` — duplicate -> early return.
    5. Fetch the URL body (static fetcher first, dynamic fallback for SPAs).
    6. Run the existing brain pipeline (triage → embed → genealogist → save).
    7. Translate brain state into an :class:`AddUrlResult`.

The return value always carries the *attempted* URL plus a status enum that
the caller can render however they like.  No exception escapes this function
under normal failure modes — fetch errors, robots blocks, and brain
exceptions are all caught and mapped to an :class:`AddUrlResult`.
"""
from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.pipeline import run_brain_pipeline
from argos.crawler._robots import RobotsDisallowed, is_robots_allowed
from argos.crawler.dynamic_fetcher import (
    _is_safe_url,
    extract_main_content,
    fetch_dynamic_page,
)
from argos.crawler.static_fetcher import _get_with_retry, _truncate_raw_content
from argos.models.tech_item import TechItem

__all__ = [
    "AddUrlResult",
    "AddUrlStatus",
    "add_url",
]

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_FETCH_TIMEOUT_SECONDS = 20.0


class AddUrlStatus(str, enum.Enum):
    """Outcome of a manual URL add attempt."""

    CREATED = "created"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class AddUrlResult:
    """Result of attempting to add a single URL.

    Attributes:
        url: The URL that was attempted (post-redirect on success).
        status: One of :class:`AddUrlStatus`.
        tech_item_id: UUID of the saved or already-present tech_item, if any.
        reason: Human-readable reason for non-CREATED outcomes.
    """

    url: str
    status: AddUrlStatus
    tech_item_id: uuid.UUID | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _find_existing_tech_item_id(
    session: AsyncSession, source_url: str
) -> uuid.UUID | None:
    """Return the tech_item id for *source_url* or None if absent."""
    result = await session.execute(
        select(TechItem.id).where(TechItem.source_url == source_url)
    )
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


async def _fetch_url_content(url: str) -> dict | None:
    """Fetch a URL and extract main content.

    Tries the lightweight static (httpx + readability) path first.  If that
    yields no usable content (empty body, non-HTML response, or transport
    error other than RobotsDisallowed), falls back to Playwright via
    :func:`fetch_dynamic_page` which handles JS-rendered SPAs.

    Returns ``{"title": str, "raw_content": str, "source_url": final_url}``
    or ``None`` if both paths fail.
    """
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        try:
            response = await _get_with_retry(client, url)
        except RobotsDisallowed:
            # Surface robots blocks to the caller — no fallback to dynamic.
            raise
        except httpx.HTTPError as exc:
            logger.info("add_url static fetch failed for %s: %r; trying dynamic", url, exc)
            response = None

    if response is not None:
        content_type = response.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type and "text" not in content_type:
            logger.info(
                "add_url static fetch non-HTML for %s (content-type=%s); trying dynamic",
                url,
                content_type,
            )
        else:
            title, body = extract_main_content(response.text)
            final_url = str(response.url) or url
            # Re-validate the post-redirect URL: httpx follows redirects
            # transparently, so the original SSRF + robots checks only cover
            # the user-supplied URL. A 30x to a private/link-local host or a
            # robots-disallowed target would otherwise slip through.
            if final_url != url:
                if not await _is_safe_url(final_url):
                    logger.warning(
                        "add_url: SSRF redirect blocked %s -> %s (failed _is_safe_url)",
                        url,
                        final_url,
                    )
                    return None
                if not await is_robots_allowed(final_url):
                    logger.warning(
                        "add_url: robots-disallowed redirect %s -> %s",
                        url,
                        final_url,
                    )
                    raise RobotsDisallowed(final_url)
            if body.strip():
                return {
                    "title": title or "",
                    "raw_content": _truncate_raw_content(body.strip()),
                    "source_url": final_url,
                }

    # Static path returned no usable content — fall back to dynamic.
    dynamic = await fetch_dynamic_page(url)
    if dynamic is None:
        return None
    return {
        "title": dynamic.get("title") or "",
        "raw_content": _truncate_raw_content((dynamic.get("raw_content") or "").strip()),
        "source_url": dynamic.get("source_url") or url,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _parse_and_validate(url: str) -> tuple[str | None, str | None]:
    """Return (cleaned_url, error_reason).

    The cleaned URL has no surrounding whitespace.  If the URL is unusable,
    returns (None, reason).
    """
    candidate = (url or "").strip()
    if not candidate:
        return None, "URL is empty"
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        return None, f"unparseable URL: {exc}"
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return None, (
            f"unsupported scheme {parts.scheme!r} — only http/https are allowed"
        )
    if not parts.netloc or not parts.hostname:
        return None, "URL is missing a host"
    return candidate, None


async def add_url(url: str, session: AsyncSession) -> AddUrlResult:
    """Add a single URL through the full crawl + brain pipeline.

    See module docstring for the step-by-step semantics.
    """
    # ── 1. Parse + scheme whitelist ───────────────────────────────────────
    cleaned, validation_error = _parse_and_validate(url)
    if cleaned is None:
        return AddUrlResult(
            url=url, status=AddUrlStatus.REJECTED, reason=validation_error
        )

    # ── 2. SSRF guard ─────────────────────────────────────────────────────
    if not await _is_safe_url(cleaned):
        return AddUrlResult(
            url=cleaned,
            status=AddUrlStatus.REJECTED,
            reason="host failed SSRF safety check (private, link-local, loopback, or unresolvable)",
        )

    # ── 3. robots.txt ─────────────────────────────────────────────────────
    if not await is_robots_allowed(cleaned):
        return AddUrlResult(
            url=cleaned,
            status=AddUrlStatus.REJECTED,
            reason="robots.txt disallows fetching this URL",
        )

    # ── 4. Dedup check ────────────────────────────────────────────────────
    existing_id = await _find_existing_tech_item_id(session, cleaned)
    if existing_id is not None:
        return AddUrlResult(
            url=cleaned,
            status=AddUrlStatus.DUPLICATE,
            tech_item_id=existing_id,
            reason="URL already present in tech_items",
        )

    # ── 5. Fetch ──────────────────────────────────────────────────────────
    try:
        fetched = await _fetch_url_content(cleaned)
    except RobotsDisallowed:
        return AddUrlResult(
            url=cleaned,
            status=AddUrlStatus.REJECTED,
            reason="robots.txt disallows fetching this URL",
        )
    except Exception as exc:  # noqa: BLE001 — fetch path covers many exception types
        logger.warning("add_url fetch raised for %s: %r", cleaned, exc)
        return AddUrlResult(
            url=cleaned,
            status=AddUrlStatus.ERROR,
            reason=f"fetch failed: {exc}",
        )

    if fetched is None:
        return AddUrlResult(
            url=cleaned,
            status=AddUrlStatus.ERROR,
            reason="fetch failed: no content retrieved",
        )

    final_url = fetched.get("source_url") or cleaned
    raw_content = (fetched.get("raw_content") or "").strip()
    if not raw_content:
        return AddUrlResult(
            url=final_url,
            status=AddUrlStatus.ERROR,
            reason="fetch returned empty content",
        )

    # If the fetcher followed a redirect, the final URL might already be in
    # the DB (different from the input URL).  Re-check dedup on final_url.
    if final_url != cleaned:
        existing_id = await _find_existing_tech_item_id(session, final_url)
        if existing_id is not None:
            return AddUrlResult(
                url=final_url,
                status=AddUrlStatus.DUPLICATE,
                tech_item_id=existing_id,
                reason="post-redirect URL already present in tech_items",
            )

    # ── 6. Brain pipeline ─────────────────────────────────────────────────
    try:
        state = await run_brain_pipeline(
            raw_text=raw_content,
            source_url=final_url,
            session=session,
        )
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        logger.warning(
            "add_url brain pipeline raised for %s: %r", final_url, exc
        )
        await _safe_rollback(session)
        return AddUrlResult(
            url=final_url,
            status=AddUrlStatus.ERROR,
            reason=f"brain pipeline error: {exc}",
        )

    # ── 7. Translate brain state to result ────────────────────────────────
    if not state.get("is_valid"):
        # Triage rejected; nothing was written.
        return AddUrlResult(
            url=final_url,
            status=AddUrlStatus.REJECTED,
            reason="triage rejected this item as not a substantive tech signal",
        )

    if not state.get("saved"):
        # is_valid=True but save_node didn't write — this is a hard error.
        return AddUrlResult(
            url=final_url,
            status=AddUrlStatus.ERROR,
            reason="brain pipeline did not persist the item",
        )

    try:
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "add_url commit failed for %s: %r", final_url, exc
        )
        await _safe_rollback(session)
        return AddUrlResult(
            url=final_url,
            status=AddUrlStatus.ERROR,
            reason=f"database commit failed: {exc}",
        )

    new_id = await _find_existing_tech_item_id(session, final_url)
    return AddUrlResult(
        url=final_url,
        status=AddUrlStatus.CREATED,
        tech_item_id=new_id,
    )


async def _safe_rollback(session: AsyncSession) -> None:
    """Best-effort rollback after a partial-pipeline failure."""
    try:
        await session.rollback()
    except Exception:  # noqa: BLE001 — best effort only
        logger.debug("add_url rollback failed", exc_info=True)
