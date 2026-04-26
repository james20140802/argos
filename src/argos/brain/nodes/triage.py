from __future__ import annotations
import json
import logging
from pydantic import BaseModel
from argos.brain.graph_state import BrainState
from argos.brain.ollama_client import SMALL_MODEL, query_ollama, unload_model

logger = logging.getLogger(__name__)

_TRIAGE_PROMPT = """Analyze the following text and determine if it describes a real technology (tool, library, framework, model, protocol, or platform).
Respond ONLY with valid JSON: {{"is_valid": true/false, "reason": "brief explanation"}}

Text:
{text}"""


class _TriageResult(BaseModel):
    is_valid: bool
    reason: str


async def triage_node(state: BrainState) -> BrainState:
    prompt = _TRIAGE_PROMPT.format(text=state["raw_text"][:2000])
    try:
        raw = await query_ollama(SMALL_MODEL, prompt, keep_alive=0)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        result = _TriageResult.model_validate_json(raw[start:end])
        return {**state, "is_valid": result.is_valid}
    except Exception as exc:
        logger.warning("triage_node failed: %r", exc)
        return {**state, "is_valid": False}
    finally:
        try:
            await unload_model(SMALL_MODEL)
        except Exception:
            pass
