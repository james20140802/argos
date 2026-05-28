from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from uuid import UUID

from sqlalchemy import update

from argos.brain.weekly_report import build_weekly_keep_report
from argos.config import settings
from argos.database import AsyncSessionLocal
from argos.models.tech_item import CategoryType, TechItem
from argos.slack.app import build_app
from argos.slack.blocks import (
    build_category_header_blocks,
    build_header_blocks,
    build_item_blocks,
    build_weekly_keep_summary_blocks,
)
from argos.slack.services.briefing_query import fetch_today_briefing

logger = logging.getLogger(__name__)

_ORDERED_CATEGORIES = (CategoryType.MAINSTREAM, CategoryType.ALPHA)


async def dispatch_daily_briefing(*, channel: str | None = None) -> str | None:
    async with AsyncSessionLocal() as session:
        now_utc = datetime.now(timezone.utc)
        items_by_category = await fetch_today_briefing(
            session,
            now_utc=now_utc,
            limit_per_category=settings.user.briefing.limit_per_category,
            topics=settings.user.interests.topics,
            lookback_days=settings.user.briefing.lookback_days,
        )

    app = build_app()
    target_channel = channel or settings.user.slack.channel_id

    if all(not items for items in items_by_category.values()):
        logger.info("No items in lookback window — posting empty-state message")
        response = await app.client.chat_postMessage(
            channel=target_channel,
            text="오늘 브리핑할 최신 소식이 없습니다",
        )
        return response.get("ts")

    header_response = await app.client.chat_postMessage(
        channel=target_channel,
        blocks=build_header_blocks(date.today(), has_items=True),
        text="Argos Daily Briefing",
    )
    header_ts: str = header_response["ts"]

    briefed_ids: list[UUID] = []
    for category in _ORDERED_CATEGORIES:
        items = items_by_category.get(category) or []
        if not items:
            continue
        await app.client.chat_postMessage(
            channel=target_channel,
            thread_ts=header_ts,
            blocks=build_category_header_blocks(category),
            text=category.value,
        )
        for item in items:
            await app.client.chat_postMessage(
                channel=target_channel,
                thread_ts=header_ts,
                blocks=build_item_blocks(item),
                text=item.title,
                unfurl_links=True,
                unfurl_media=True,
            )
            briefed_ids.append(item.id)

    if briefed_ids:
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(TechItem)
                .where(TechItem.id.in_(briefed_ids))
                .values(briefed_at=now_utc)
            )
            await session.commit()

    return header_ts


async def dispatch_weekly_briefing(*, channel: str | None = None) -> str | None:
    """Dispatch the weekly Keep portfolio summary to Slack (ARG-123).

    Unlike :func:`dispatch_daily_briefing`, this sends exactly **one** message
    (no thread, no per-item follow-ups).  An empty Keep portfolio still
    triggers a placeholder message — spec mandates "skip 금지".

    Parameters
    ----------
    channel:
        Optional override for the target channel ID.  Defaults to
        ``settings.user.slack.channel_id``.

    Returns
    -------
    str | None
        The ``ts`` of the posted message, or ``None`` if Slack returned a
        response without a ``ts`` field (defensive).
    """
    async with AsyncSessionLocal() as session:
        report = await build_weekly_keep_report(session)

    app = build_app()
    target_channel = channel or settings.user.slack.channel_id

    blocks = build_weekly_keep_summary_blocks(report)
    start_label = report.window_start.date().isoformat()
    end_label = report.window_end.date().isoformat()
    fallback = f"Weekly Keep 현황 ({start_label} ~ {end_label})"

    response = await app.client.chat_postMessage(
        channel=target_channel,
        blocks=blocks,
        text=fallback,
    )
    return response.get("ts")
