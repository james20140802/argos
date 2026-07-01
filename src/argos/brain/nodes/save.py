from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.graph_state import BrainState
from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import RelationType, TechSuccession

logger = logging.getLogger(__name__)

_RELATION_MAP: dict[str, RelationType] = {
    "Replace": RelationType.REPLACE,
    "Enhance": RelationType.ENHANCE,
    "Fork": RelationType.FORK,
}


async def save_node(
    state: BrainState, session: AsyncSession, *, flush: bool = True
) -> BrainState:
    """Persist a BrainState to the database.

    Parameters
    ----------
    flush:
        When ``True`` (default) an explicit ``await session.flush()`` is issued
        after adding the item, and ``state["saved"]`` is set to ``True`` only
        after that flush succeeds.  Pass ``flush=False`` in the batch pipeline
        so save_node does not flush and does not set ``saved=True``; the caller
        must flush inside a savepoint and set ``saved["saved"] = True`` only
        after the flush succeeds, ensuring a failed flush leaves the state with
        ``saved=False`` for correct retry handling.

        Note: TechItem.id is pre-assigned via ``uuid.uuid4()`` in the
        constructor, so ``saved_item_id`` and succession FKs are available
        regardless of whether flush was called.

    Autoflush caveat
    ----------------
    The session factory (``database.py``) leaves ``autoflush=True`` (SQLAlchemy
    default).  This means each ``session.execute(SELECT ...)`` call inside this
    function — e.g. the duplicate-URL check and the predecessor existence check
    — can still trigger an implicit flush for any pending items.  Passing
    ``flush=False`` eliminates the *explicit* per-item flush, reducing round-trips
    from N to 1 at the batch level, but does not suppress autoflush-triggered
    flushes during in-function SELECT queries.  This is an acceptable trade-off
    for the batch pipeline; callers that need strict flush control should wrap
    the session in a ``with session.no_autoflush:`` block.
    """
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

    # Pre-assign the PK so it is available for saved_item_id and succession FK
    # even when flush=False (SQLAlchemy populates callable defaults at flush
    # time, not at object construction, so we assign explicitly here).
    item = TechItem(
        id=uuid.uuid4(),
        title=title,
        source_url=state["source_url"],
        raw_content=state["raw_text"],
        summary=state.get("summary"),
        digest=state.get("digest"),
        # Use triage-decided category, falling back to ALPHA as a safe default
        # in case it was not set (e.g. state produced by an older code path).
        category=state.get("category") or CategoryType.ALPHA,
        trust_score=state.get("trust_score"),
        published_at=state.get("published_at"),
        image_url=state.get("image_url"),
    )

    extracted_info = state.get("extracted_info") or {}
    if "embedding" in extracted_info:
        item.embedding = extracted_info["embedding"]

    session.add(item)
    if flush:
        await session.flush()
        state["saved"] = True
    # Surface the new item's PK so downstream stages (ARG-103: succession
    # alerts) can collect just the freshly-saved IDs without re-querying.
    # Pre-assigned via uuid.uuid4() so this is available regardless of flush.
    state["saved_item_id"] = item.id

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
                predecessor_uuid = None
            if predecessor_uuid is not None:
                predecessor_exists = await session.execute(
                    select(TechItem.id).where(TechItem.id == predecessor_uuid)
                )
                if predecessor_exists.scalar_one_or_none() is None:
                    logger.warning(
                        "save_node: predecessor %s not found in DB, skipping succession",
                        predecessor_uuid,
                    )
                else:
                    succession = TechSuccession(
                        predecessor_id=predecessor_uuid,
                        successor_id=item.id,
                        relation_type=mapped_enum,
                        reasoning=succession_result.get("reason", ""),
                    )
                    session.add(succession)

    return state
