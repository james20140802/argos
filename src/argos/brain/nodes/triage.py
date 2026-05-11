from __future__ import annotations
import logging
from pydantic import BaseModel, StrictBool, field_validator
from argos.brain.graph_state import BrainState
from argos.brain.llm_client import get_llm_client
from argos.config import settings

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 500
_MAX_INTERESTS = 10
_INTEREST_TRUST_BUMP = 0.1
_TERM_MAX_CHARS = 64
_TRIAGE_TEXT_MAX_CHARS = 2000

_TRIAGE_PROMPT = """Analyze the following text and determine if it describes a real technology (tool, library, framework, model, protocol, or platform).
trust_score reflects substance over hype: 0.0=pure marketing, 0.5=neutral, 1.0=well-evidenced technical detail.
summary is a 1-2 sentence factual blurb (max 500 chars) describing what the technology is and why it matters; written in {language}. Use null if is_valid is false.
Respond ONLY with valid JSON: {{"is_valid": true/false, "reason": "brief explanation", "trust_score": 0.0-1.0, "summary": "..."}}
{interests_block}
Text:
{text}"""


class _TriageResult(BaseModel):
    is_valid: StrictBool
    reason: str
    trust_score: float | None = None
    summary: str | None = None

    @field_validator("trust_score", mode="before")
    @classmethod
    def _normalize_trust(cls, v):
        if v is None or isinstance(v, bool):
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

    @field_validator("summary", mode="before")
    @classmethod
    def _normalize_summary(cls, v):
        if v is None or isinstance(v, bool):
            return None
        if not isinstance(v, str):
            return None
        s = v.strip()
        if not s or s.lower() in {"null", "none"}:
            return None
        return s[:_SUMMARY_MAX_CHARS]


def _sanitize_term(t: str) -> str:
    # Strip control chars, newlines, and quotes to defang prompt-injection from config.
    cleaned = "".join(ch for ch in t if ch.isprintable() and ch not in "\n\r\t\"'`")
    cleaned = cleaned.strip()
    return cleaned[:_TERM_MAX_CHARS]


def _normalize_terms(raw: list) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        term = _sanitize_term(item)
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    original = len(out)
    if original > _MAX_INTERESTS:
        logger.warning("triage interests truncated: %d -> %d", original, _MAX_INTERESTS)
        out = out[:_MAX_INTERESTS]
    return out


def _build_interests_block(topics: list[str], exclusions: list[str]) -> str:
    if not topics and not exclusions:
        return ""
    lines: list[str] = []
    if topics:
        lines.append(
            "User interests (boost relevance if matched): " + ", ".join(topics)
        )
    if exclusions:
        lines.append(
            "Exclusions (pass immediately if matched): " + ", ".join(exclusions)
        )
    return "\n".join(lines)


def _apply_interest_rules(
    triage_text: str,
    result: _TriageResult,
    topics: list[str],
    exclusions: list[str],
) -> tuple[bool, float | None, str | None]:
    # NOTE: substring matching is intentional for v1; a word-boundary regex is a
    # follow-up if false positives (e.g. "crypto" matching "cryptography") prove noisy.
    # triage_text must be the same truncated window passed to the LLM so deterministic
    # rules and the model decision stay consistent (see ARG-50 review).
    haystack = (triage_text or "") + " " + (result.summary or "")
    haystack_lower = haystack.lower()

    for term in exclusions:
        if term and term.lower() in haystack_lower:
            logger.info("triage exclusion hit: %s", term)
            return (False, 0.0, None)

    if result.is_valid and result.trust_score is not None:
        for term in topics:
            if term and term.lower() in haystack_lower:
                bumped = min(1.0, result.trust_score + _INTEREST_TRUST_BUMP)
                summary = result.summary if result.is_valid else None
                return (result.is_valid, bumped, summary)

    summary = result.summary if result.is_valid else None
    return (result.is_valid, result.trust_score, summary)


async def triage_node(state: BrainState) -> BrainState:
    topics = _normalize_terms(settings.user.interests.topics)
    exclusions = _normalize_terms(settings.user.interests.exclusions)
    interests_block = _build_interests_block(topics, exclusions)
    triage_text = (state["raw_text"] or "")[:_TRIAGE_TEXT_MAX_CHARS]
    prompt = _TRIAGE_PROMPT.format(
        text=triage_text,
        language=settings.user.slack.summary_language,
        interests_block=interests_block,
    )
    client = get_llm_client()
    try:
        raw = await client.query("small", prompt, keep_alive=0)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        result = _TriageResult.model_validate_json(raw[start:end])

        if topics or exclusions:
            is_valid, trust_score, summary = _apply_interest_rules(
                triage_text, result, topics, exclusions
            )
        else:
            is_valid = result.is_valid
            trust_score = result.trust_score
            summary = result.summary if result.is_valid else None

        return {
            **state,
            "is_valid": is_valid,
            "trust_score": trust_score,
            "summary": summary,
        }
    except Exception as exc:
        logger.warning("triage_node failed: %r", exc)
        return {**state, "is_valid": False, "trust_score": None, "summary": None}
    finally:
        try:
            await client.unload("small")
        except Exception:
            pass
