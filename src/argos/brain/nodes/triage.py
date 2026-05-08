from __future__ import annotations
import json
import logging
from pydantic import BaseModel, StrictBool, field_validator
from argos.brain.graph_state import BrainState
from argos.brain.ollama_client import SMALL_MODEL, query_ollama, unload_model

logger = logging.getLogger(__name__)

_TRIAGE_PROMPT = """Analyze the following text and determine if it describes a real technology (tool, library, framework, model, protocol, or platform).
trust_score reflects substance over hype: 0.0=pure marketing, 0.5=neutral, 1.0=well-evidenced technical detail.
Respond ONLY with valid JSON: {{"is_valid": true/false, "reason": "brief explanation", "trust_score": 0.0-1.0}}

Text:
{text}"""


class _TriageResult(BaseModel):
    is_valid: StrictBool
    reason: str
    trust_score: float | None = None

    @field_validator("trust_score", mode="before")
    @classmethod
    def _normalize_trust(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"", "null", "none"}:
                return None
            try:
                v = float(s)
            except ValueError:
                return None
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return None


async def triage_node(state: BrainState) -> BrainState:
    prompt = _TRIAGE_PROMPT.format(text=state["raw_text"][:2000])
    try:
        raw = await query_ollama(SMALL_MODEL, prompt, keep_alive=0)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        result = _TriageResult.model_validate_json(raw[start:end])
        return {**state, "is_valid": result.is_valid, "trust_score": result.trust_score}
    except Exception as exc:
        logger.warning("triage_node failed: %r", exc)
        return {**state, "is_valid": False, "trust_score": None}
    finally:
        try:
            await unload_model(SMALL_MODEL)
        except Exception:
            pass
