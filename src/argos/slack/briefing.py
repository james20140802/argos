from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from argos.config import settings
from argos.database import AsyncSessionLocal
from argos.slack.app import build_app
from argos.slack.blocks import build_briefing_blocks
from argos.slack.services.briefing_query import fetch_today_briefing

logger = logging.getLogger(__name__)


async def dispatch_daily_briefing(*, channel: str | None = None) -> str | None:
    async with AsyncSessionLocal() as session:
        now_utc = datetime.now(timezone.utc)
        items_by_category = await fetch_today_briefing(session, now_utc=now_utc)

    if all(not items for items in items_by_category.values()):
        logger.info("No items today — skipping briefing dispatch")
        return None

    app = build_app()
    blocks = build_briefing_blocks(items_by_category, today=date.today())
    target_channel = channel or settings.SLACK_CHANNEL_ID
    response = await app.client.chat_postMessage(
        channel=target_channel,
        blocks=blocks,
        text="Argos Daily Briefing",
    )
    ts: str = response["ts"]
    return ts
