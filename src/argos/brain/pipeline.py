from __future__ import annotations
import asyncio
import contextlib

from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node
from argos.brain.nodes.embed import embed_and_search_node
from argos.brain.nodes.genealogist import genealogist_node
from argos.brain.nodes.save import save_node
from argos.brain.llm_client import get_llm_client
from argos.models.tech_item import CategoryType


async def run_brain_pipeline(
    raw_text: str,
    source_url: str,
    session: AsyncSession,
    *,
    source_category: CategoryType | None = None,
) -> BrainState:
    # source_category is an optional hint from the fetcher (e.g. RSS in ARG-52,
    # arXiv in ARG-53) indicating which category the source leans towards.
    # GitHub/HN fetchers do not pass it (defaults to None).
    # Callers in run_full_pipeline may forward item.get("source_category") here
    # once ARG-52/53 land; the field is ignored by current GitHub/HN paths.
    initial: BrainState = {
        "raw_text": raw_text,
        "source_url": source_url,
        "is_valid": False,
        "trust_score": None,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": source_category,
        "category": None,
    }
    triaged = await triage_node(initial)
    if not triaged["is_valid"]:
        return triaged

    # Run embed_and_search first so we can decide whether to spend VRAM on the
    # 32B prewarm. On cold start the genealogist branch is skipped and we never
    # need to load the large model.
    embedded = await embed_and_search_node(triaged, session=session)
    if embedded.get("genealogy_skipped"):
        return await save_node(embedded, session=session)

    prewarm_task = asyncio.create_task(get_llm_client().prewarm("large"))
    try:
        genealogized = await genealogist_node(embedded, prewarm_task=prewarm_task)
        return await save_node(genealogized, session=session)
    finally:
        if not prewarm_task.done():
            prewarm_task.cancel()
        with contextlib.suppress(BaseException):
            await prewarm_task
