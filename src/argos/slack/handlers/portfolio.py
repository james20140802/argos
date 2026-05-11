from __future__ import annotations

import logging

from argos.database import AsyncSessionLocal
from argos.slack.blocks import build_portfolio_blocks, build_portfolio_empty_blocks
from argos.slack.services.briefing_query import fetch_user_portfolio

logger = logging.getLogger(__name__)


async def _render_portfolio(respond) -> None:
    """포트폴리오를 조회하여 ephemeral 메시지로 응답하는 공유 헬퍼."""
    async with AsyncSessionLocal() as session:
        assets = await fetch_user_portfolio(session)

    if assets:
        blocks = build_portfolio_blocks(assets)
    else:
        blocks = build_portfolio_empty_blocks()

    await respond(
        blocks=blocks,
        text="내 포트폴리오",
        response_type="ephemeral",
        replace_original=False,
    )


async def handle_portfolio_command(ack, command, respond) -> None:
    """/argos slash command 핸들러 — 포트폴리오를 ephemeral로 표시한다."""
    await ack()
    await _render_portfolio(respond)


async def handle_portfolio_mention(event, say, respond=None) -> None:
    """app_mention 이벤트에서 'portfolio' 키워드를 처리한다."""
    text: str = (event.get("text") or "").lower()
    if "portfolio" not in text:
        return
    if respond is not None:
        await _render_portfolio(respond)
    else:
        async with AsyncSessionLocal() as session:
            assets = await fetch_user_portfolio(session)

        if assets:
            blocks = build_portfolio_blocks(assets)
        else:
            blocks = build_portfolio_empty_blocks()

        await say(blocks=blocks, text="내 포트폴리오")
