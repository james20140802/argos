"""Slack `/argos add <URL>` slash command handler (ARG-110).

This module replaces the previous direct `/argos` → portfolio binding with
a dispatcher.  When the user types ``/argos add <URL> [URL...]`` we route to
:func:`_handle_add`; everything else falls back to the existing portfolio
handler so the legacy ``/argos`` (no args) UX still works.
"""
from __future__ import annotations

import asyncio
import logging
import re

from argos.crawler.add_url import add_url as add_url_service
from argos.database import AsyncSessionLocal
from argos.slack.blocks import (
    build_add_url_help_blocks,
    build_add_url_processing_blocks,
    build_add_url_result_blocks,
)
from argos.slack.handlers.portfolio import handle_portfolio_command

__all__ = [
    "handle_argos_slash_command",
    "parse_add_command",
]

logger = logging.getLogger(__name__)

# Strip Slack auto-link wrappers: <url> or <url|displayed>.
_SLACK_LINK_RE = re.compile(r"^<([^|>]+)(?:\|[^>]*)?>$")


def parse_add_command(text: str) -> list[str]:
    """Return the URLs from a ``/argos add <URL> [URL...]`` command body.

    Returns an empty list if the command is not an ``add`` invocation or if
    no URLs follow the keyword.  Strips Slack's ``<...>`` auto-link wrapper.
    """
    if not text:
        return []
    tokens = text.strip().split()
    if not tokens:
        return []
    if tokens[0].lower() != "add":
        return []
    urls: list[str] = []
    for tok in tokens[1:]:
        match = _SLACK_LINK_RE.match(tok)
        urls.append(match.group(1) if match else tok)
    return urls


async def _process_add_urls(urls: list[str], respond) -> None:
    """Background task: invoke add_url_service for each URL, then reply.

    Errors are caught and surfaced as a friendly user-facing message; the
    full traceback is logged for the operator.
    """
    try:
        async with AsyncSessionLocal() as session:
            results = []
            for url in urls:
                result = await add_url_service(url, session)
                results.append(result)
    except Exception as exc:
        logger.exception("argos add background task failed: %r", exc)
        await respond(
            text="URL 추가 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            response_type="ephemeral",
            replace_original=False,
        )
        return

    blocks = build_add_url_result_blocks(results)
    await respond(
        blocks=blocks,
        text="URL 처리 결과",
        response_type="ephemeral",
        replace_original=False,
    )


async def _handle_add(urls: list[str], respond) -> None:
    """Send the "processing…" interim message and kick off the background work."""
    if not urls:
        await respond(
            blocks=build_add_url_help_blocks(),
            text="URL을 입력해 주세요",
            response_type="ephemeral",
            replace_original=False,
        )
        return

    # Interim "processing" message so the user knows the command was accepted.
    await respond(
        blocks=build_add_url_processing_blocks(urls),
        text=f"⏳ {len(urls)}개 URL 처리 중…",
        response_type="ephemeral",
        replace_original=False,
    )

    # Spawn the actual work; ack must already have returned.
    asyncio.create_task(_process_add_urls(urls, respond))


async def handle_argos_slash_command(ack, command, respond) -> None:
    """Dispatcher for the ``/argos`` slash command.

    Routes ``add <URL>...`` to the URL-injection flow.  Anything else (the
    bare ``/argos`` invocation) goes to the legacy portfolio handler.
    """
    text = (command.get("text") or "").strip()
    tokens = text.split()
    if tokens and tokens[0].lower() == "add":
        await ack()
        urls = parse_add_command(text)
        await _handle_add(urls, respond)
        return

    # Fallback: delegate to portfolio (which handles its own ack()).
    await handle_portfolio_command(ack, command, respond)
