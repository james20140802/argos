from __future__ import annotations
import logging
from typing import Callable
from pydantic import BaseModel, StrictBool, field_validator
from argos.brain._language import language_directive
from argos.brain.graph_state import BrainState
from argos.brain.llm_client import get_llm_client
from argos.config import settings
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 500
_MAX_INTERESTS = 10
_INTEREST_TRUST_BUMP = 0.1
_TERM_MAX_CHARS = 64
_TRIAGE_TEXT_MAX_CHARS = 2000

_TRIAGE_PROMPT = """Analyze the following text and determine if it describes a real technology (tool, library, framework, model, protocol, or platform).
trust_score reflects substance over hype: 0.0=pure marketing, 0.5=neutral, 1.0=well-evidenced technical detail.
reason is a brief 1-sentence justification of the is_valid and category decision; written in {language}.
summary is a 1-2 sentence factual blurb (max 500 chars) describing what the technology is and why it matters; written in {language}. Use null if is_valid is false.
category must be one of "Mainstream" or "Alpha". Mainstream = mature, widely adopted technology; Alpha = cutting-edge, experimental, or niche. Default to "Alpha" when uncertain.
{source_hint_block}Respond ONLY with valid JSON: {schema}
{interests_block}
Text:
{text}{language_reminder}"""

_SCHEMA_BASE = '{{"is_valid": true/false, "reason": "brief explanation", "trust_score": 0.0-1.0, "summary": "...", "category": "Mainstream"|"Alpha"}}'
_SCHEMA_WITH_RELEVANCE = '{{"is_valid": true/false, "reason": "brief explanation", "trust_score": 0.0-1.0, "summary": "...", "category": "Mainstream"|"Alpha", "is_relevant": true/false}}'


class _TriageResult(BaseModel):
    is_valid: StrictBool
    reason: str
    trust_score: float | None = None
    summary: str | None = None
    is_relevant: StrictBool = True
    category: CategoryType = CategoryType.ALPHA

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v):
        """Accept case-insensitive strings; fall back to ALPHA on null/garbage."""
        if v is None:
            return CategoryType.ALPHA
        if isinstance(v, CategoryType):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in {"null", "none"}:
                return CategoryType.ALPHA
            # Case-insensitive match against enum values ("Mainstream", "Alpha")
            for member in CategoryType:
                if s.lower() == member.value.lower():
                    return member
        return CategoryType.ALPHA

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


def _build_source_hint_block(source_category: CategoryType | str | None) -> str:
    """Return an optional one-line hint for the LLM, or empty string.

    Accepts a ``CategoryType`` enum member or a plain string (e.g. ``"Mainstream"``
    as may be provided by fetcher-supplied item dicts).  Unrecognised or null values
    produce an empty string rather than raising ``AttributeError``.
    """
    if source_category is None:
        return ""
    if isinstance(source_category, CategoryType):
        label = source_category.value
    elif isinstance(source_category, str):
        s = source_category.strip()
        matched: CategoryType | None = None
        for member in CategoryType:
            if s.lower() == member.value.lower():
                matched = member
                break
        if matched is None:
            return ""
        label = matched.value
    else:
        return ""
    return (
        f"Source hint: this item came from a {label}-leaning source;"
        " weigh accordingly but rely on content for the final decision.\n"
    )


def _build_interests_block(topics: list[str], exclusions: list[str]) -> str:
    if not topics and not exclusions:
        return ""
    lines: list[str] = []
    if topics:
        lines.append(
            "User interests (boost relevance if matched): " + ", ".join(topics)
        )
        lines.append(
            "Is this text directly related to one of the user interest topics above?"
            " Set is_relevant to true if yes, false if no."
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


async def _triage_one(state: BrainState, client, keep_alive) -> BrainState:
    """Run triage for a single state without managing model load/unload."""
    topics = _normalize_terms(settings.user.interests.topics)
    exclusions = _normalize_terms(settings.user.interests.exclusions)
    interests_block = _build_interests_block(topics, exclusions)
    source_hint_block = _build_source_hint_block(state.get("source_category"))
    schema = _SCHEMA_WITH_RELEVANCE if topics else _SCHEMA_BASE
    triage_text = (state["raw_text"] or "")[:_TRIAGE_TEXT_MAX_CHARS]
    _language = settings.user.slack.summary_language or "English"
    prompt = _TRIAGE_PROMPT.format(
        text=triage_text,
        language=_language,
        interests_block=interests_block,
        source_hint_block=source_hint_block,
        schema=schema,
        language_reminder=language_directive(_language),
    )
    try:
        raw = await client.query(
            "small", prompt, keep_alive=keep_alive, num_ctx=settings.user.triage.num_ctx
        )
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        result = _TriageResult.model_validate_json(raw[start:end])

        if topics and not result.is_relevant:
            logger.info(
                "triage relevance gate: is_relevant=False for topics=%s; demoting to invalid",
                topics,
            )
            is_valid = False
            trust_score = None
            summary = None
            category = None
        elif topics or exclusions:
            is_valid, trust_score, summary = _apply_interest_rules(
                triage_text, result, topics, exclusions
            )
            category = result.category if is_valid else None
        else:
            is_valid = result.is_valid
            trust_score = result.trust_score
            summary = result.summary if result.is_valid else None
            category = result.category if result.is_valid else None

        return {
            **state,
            "is_valid": is_valid,
            "trust_score": trust_score,
            "summary": summary,
            "category": category,
        }
    except Exception as exc:
        logger.warning("triage_node failed: %r", exc)
        return {
            **state,
            "is_valid": False,
            "trust_score": None,
            "summary": None,
            "category": None,
        }


async def triage_node(state: BrainState) -> BrainState:
    client = get_llm_client()
    try:
        return await _triage_one(state, client, keep_alive=0)
    finally:
        try:
            await client.unload("small")
        except Exception:
            pass


async def batch_triage_states(
    states: list[BrainState],
    *,
    on_item_done: Callable[[], None] | None = None,
) -> list[BrainState]:
    """Triage all states with the 8B model loaded once across all items.

    The model is kept alive (keep_alive='5m') for every call and unloaded
    once after all items are processed — reducing N model swaps to 1.

    Parameters
    ----------
    states:
        Input batch of brain states to triage.
    on_item_done:
        Optional zero-arg callback invoked once after each state finishes
        triage (success or failure). Provided so the CLI can drive a Rich
        progress bar (ARG-92/ARG-101) without leaking UI concerns into the
        brain module. Defaults to ``None`` (no-op), keeping existing callers
        unaffected. Exceptions raised by the callback are swallowed so a
        broken UI cannot abort the pipeline.
    """
    if not states:
        return []
    client = get_llm_client()
    results: list[BrainState] = []
    try:
        for state in states:
            result = await _triage_one(state, client, keep_alive="5m")
            results.append(result)
            if on_item_done is not None:
                try:
                    on_item_done()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("batch_triage_states on_item_done raised: %r", exc)
    finally:
        try:
            await client.unload("small")
        except Exception:
            pass
    return results
