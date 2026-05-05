from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from argos.database import AsyncSessionLocal
from argos.models.user_asset import AssetStatus, UserAsset

logger = logging.getLogger(__name__)


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
        result = await session.execute(
            select(UserAsset).where(UserAsset.tech_id == tech_id)
        )
        asset = result.scalar_one_or_none()
        if asset is None:
            asset = UserAsset(tech_id=tech_id, status=AssetStatus.ARCHIVED)
            session.add(asset)
        else:
            asset.status = AssetStatus.ARCHIVED
        await session.commit()

    await respond(
        "⏭️ 패스했습니다. (Archived)",
        response_type="ephemeral",
        replace_original=False,
    )
