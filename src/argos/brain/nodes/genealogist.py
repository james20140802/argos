from __future__ import annotations

import asyncio
import contextlib
import logging

from pydantic import BaseModel

from argos.brain.graph_state import BrainState
from argos.brain.ollama_client import LARGE_MODEL, LARGE_MODEL_TIMEOUT, query_ollama

logger = logging.getLogger(__name__)

_GENEALOGIST_PROMPT = """You are a technology genealogist. Analyze whether the new technology replaces, enhances, or forks any of the existing technologies listed.

New technology:
{new_tech}

Existing related technologies:
{existing_techs}

Respond ONLY with valid JSON:
{{"replace_target_id": "UUID string or null", "relation_type": "Replace or Enhance or Fork or null", "reason": "brief explanation"}}"""


class _SuccessionResult(BaseModel):
    replace_target_id: str | None
    relation_type: str | None
    reason: str


async def genealogist_node(
    state: BrainState, *, prewarm_task: asyncio.Task | None = None
) -> BrainState:
    if not state["is_valid"] or not state["related_tech_ids"]:
        return state
    similar_items = (state.get("extracted_info") or {}).get("similar_items", [])
    if not similar_items:
        return state
    existing_techs = "\n".join(
        f"- ID: {item['id']}, Title: {item['title']}: {item['raw_content'][:300]}"
        for item in similar_items
    )
    prompt = _GENEALOGIST_PROMPT.format(
        new_tech=state["raw_text"][:1000],
        existing_techs=existing_techs,
    )
    try:
        if prewarm_task is not None:
            with contextlib.suppress(Exception):
                await prewarm_task
        raw = await query_ollama(
            LARGE_MODEL,
            prompt,
            keep_alive="5m",
            timeout=LARGE_MODEL_TIMEOUT,
            think=False,
        )
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON in response")
        result = _SuccessionResult.model_validate_json(raw[start:end])
        return {**state, "succession_result": result.model_dump()}
    except Exception as exc:
        logger.warning("genealogist_node failed: %r", exc)
        return {**state, "succession_result": None}
