from __future__ import annotations
import asyncio
import contextlib
import functools
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node
from argos.brain.nodes.embed import embed_and_search_node
from argos.brain.nodes.genealogist import genealogist_node
from argos.brain.nodes.save import save_node
from argos.brain.ollama_client import LARGE_MODEL, prewarm_model


def _build_post_triage_graph(
    session: AsyncSession, prewarm_task: asyncio.Task | None = None
):
    graph = StateGraph(BrainState)
    graph.add_node("embed_search", functools.partial(embed_and_search_node, session=session))
    graph.add_node(
        "genealogist", functools.partial(genealogist_node, prewarm_task=prewarm_task)
    )
    graph.add_node("save", functools.partial(save_node, session=session))
    graph.set_entry_point("embed_search")
    graph.add_edge("embed_search", "genealogist")
    graph.add_edge("genealogist", "save")
    graph.add_edge("save", END)
    return graph.compile()


async def run_brain_pipeline(
    raw_text: str, source_url: str, session: AsyncSession
) -> BrainState:
    initial: BrainState = {
        "raw_text": raw_text,
        "source_url": source_url,
        "is_valid": False,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
    }
    triaged = await triage_node(initial)
    if not triaged["is_valid"]:
        return triaged
    prewarm_task = asyncio.create_task(prewarm_model(LARGE_MODEL))
    try:
        compiled = _build_post_triage_graph(session, prewarm_task=prewarm_task)
        return await compiled.ainvoke(triaged)
    finally:
        if not prewarm_task.done():
            prewarm_task.cancel()
        with contextlib.suppress(BaseException):
            await prewarm_task
