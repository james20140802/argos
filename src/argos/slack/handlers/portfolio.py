from __future__ import annotations

import logging

from argos.database import AsyncSessionLocal
from argos.slack.blocks import build_portfolio_blocks, build_portfolio_empty_blocks
from argos.slack.services.briefing_query import fetch_user_portfolio

logger = logging.getLogger(__name__)


async def _load_portfolio_blocks() -> list[dict]:
    """포트폴리오 자산을 조회해 Block Kit 블록 목록을 빌드한다."""
    async with AsyncSessionLocal() as session:
        assets = await fetch_user_portfolio(session)

    if assets:
        return build_portfolio_blocks(assets)
    return build_portfolio_empty_blocks()


async def handle_portfolio_command(ack, command, respond) -> None:
    """/argos slash command 핸들러 — 포트폴리오를 ephemeral로 표시한다."""
    await ack()
    blocks = await _load_portfolio_blocks()
    await respond(
        blocks=blocks,
        text="내 포트폴리오",
        response_type="ephemeral",
        replace_original=False,
    )


async def handle_portfolio_mention(event, say) -> None:
    """app_mention 이벤트에서 'portfolio' 키워드를 처리한다.

    app_mention 이벤트에는 ``response_url``이 없어 ``respond``를 신뢰할 수
    없으므로 항상 ``say()``로 채널 메시지를 전송한다.
    """
    text: str = (event.get("text") or "").lower()
    if "portfolio" not in text:
        return
    blocks = await _load_portfolio_blocks()
    await say(blocks=blocks, text="내 포트폴리오")
