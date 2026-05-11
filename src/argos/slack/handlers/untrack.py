from __future__ import annotations

import logging
import uuid

from argos.database import AsyncSessionLocal
from argos.models.user_asset import AssetStatus
from argos.slack.blocks import build_portfolio_blocks, build_portfolio_empty_blocks
from argos.slack.services.asset_transition import transition_asset
from argos.slack.services.briefing_query import fetch_user_portfolio

logger = logging.getLogger(__name__)


async def handle_untrack(ack, body, respond) -> None:
    """action_untrack 버튼 핸들러 — 자산을 Archived로 전이하고 포트폴리오를 재렌더링한다."""
    await ack()

    tech_id_str: str = body["actions"][0]["value"]
    try:
        tech_id = uuid.UUID(tech_id_str)
    except ValueError:
        await respond(
            "잘못된 tech_id입니다.",
            response_type="ephemeral",
            replace_original=False,
        )
        return

    async with AsyncSessionLocal() as session:
        await transition_asset(session, tech_id, AssetStatus.ARCHIVED)
        await session.commit()
        # Re-fetch after commit so the updated list (without the untracked item) is fresh.
        assets = await fetch_user_portfolio(session)

    if assets:
        blocks = build_portfolio_blocks(assets)
    else:
        blocks = build_portfolio_empty_blocks()

    await respond(
        blocks=blocks,
        text="내 포트폴리오",
        response_type="ephemeral",
        replace_original=True,
    )
