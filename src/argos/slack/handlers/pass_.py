from __future__ import annotations

import logging
import uuid

from argos.database import AsyncSessionLocal
from argos.models.user_asset import AssetStatus
from argos.slack.blocks import upsert_item_status_block
from argos.slack.services.asset_transition import TransitionOutcome, transition_asset

logger = logging.getLogger(__name__)

_MESSAGES = {
    TransitionOutcome.CREATED: "⏭️ 패스했습니다. (Archived)",
    TransitionOutcome.TRANSITIONED: "⏭️ 상태를 Archived로 변경했습니다.",
    TransitionOutcome.NOOP: "ℹ️ 이미 Archived 상태입니다.",
}


async def handle_pass(ack, body, respond):
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
        outcome = await transition_asset(session, tech_id, AssetStatus.ARCHIVED)
        await session.commit()

    original_blocks = (body.get("message") or {}).get("blocks") or []
    new_blocks = upsert_item_status_block(original_blocks, AssetStatus.ARCHIVED)
    await respond(
        blocks=new_blocks,
        text=_MESSAGES[outcome],
        replace_original=True,
    )
