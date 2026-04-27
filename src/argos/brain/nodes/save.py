from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

logger = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.graph_state import BrainState
from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import RelationType, TechSuccession

_RELATION_MAP: dict[str, RelationType] = {
    "Replace": RelationType.REPLACE,
    "Enhance": RelationType.ENHANCE,
    "Fork": RelationType.FORK,
}


async def save_node(state: BrainState, session: AsyncSession) -> BrainState:
    if not state["is_valid"]:
        return state

    if not state["source_url"]:
        logger.warning("save_node: empty source_url, skipping")
        return state

    title = next(
        (line.strip() for line in state["raw_text"].splitlines() if line.strip()),
        "Untitled",
    )[:500]

    existing = await session.execute(
        select(TechItem.id).where(TechItem.source_url == state["source_url"])
    )
    if existing.scalar_one_or_none() is not None:
        return state

    item = TechItem(
        title=title,
        source_url=state["source_url"],
        raw_content=state["raw_text"],
        category=CategoryType.ALPHA,
    )

    extracted_info = state.get("extracted_info") or {}
    if "embedding" in extracted_info:
        item.embedding = extracted_info["embedding"]

    session.add(item)
    await session.flush()

    succession_result = state.get("succession_result")
    if succession_result is not None and succession_result.get("replace_target_id") is not None:
        relation_str = succession_result.get("relation_type")
        mapped_enum = _RELATION_MAP.get(relation_str) if relation_str else None
        if relation_str and mapped_enum is None:
            logger.warning("save_node: unrecognized relation_type %r, skipping succession", relation_str)
        if mapped_enum is not None:
            try:
                predecessor_uuid = uuid.UUID(succession_result["replace_target_id"])
            except (ValueError, AttributeError):
                logger.warning(
                    "save_node: invalid replace_target_id UUID %r, skipping succession",
                    succession_result["replace_target_id"],
                )
                return state
            succession = TechSuccession(
                predecessor_id=predecessor_uuid,
                successor_id=item.id,
                relation_type=mapped_enum,
                reasoning=succession_result.get("reason", ""),
            )
            session.add(succession)

    return state
