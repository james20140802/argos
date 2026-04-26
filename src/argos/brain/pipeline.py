from __future__ import annotations
import functools
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node
from argos.brain.nodes.embed import embed_and_search_node
from argos.brain.nodes.genealogist import genealogist_node
from argos.brain.nodes.save import save_node


def _build_graph(session: AsyncSession):
    graph = StateGraph(BrainState)
    graph.add_node("triage", triage_node)
    graph.add_node("embed_search", functools.partial(embed_and_search_node, session=session))
    graph.add_node("genealogist", genealogist_node)
    graph.add_node("save", functools.partial(save_node, session=session))
    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage",
        lambda s: "embed_search" if s["is_valid"] else END,
    )
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
    compiled = _build_graph(session)
    return await compiled.ainvoke(initial)
