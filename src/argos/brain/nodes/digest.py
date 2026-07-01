from __future__ import annotations

import logging
import re
from typing import Callable

from argos.brain._language import language_directive
from argos.brain.graph_state import BrainState
from argos.brain.llm_client import OllamaClient, get_digest_llm_client
from argos.brain.ollama_client import LARGE_MODEL_TIMEOUT
from argos.config import settings

logger = logging.getLogger(__name__)

_DIGEST_PROMPT = """You are a technology editor. Write a faithful long-form digest of the source text below, in 3–5 short paragraphs.

Rules:
- Use ONLY facts present in the source text. Do NOT invent details, numbers, or claims.
- Do NOT repeat the one-line TL;DR verbatim; expand on it.
- Plain prose only. No markdown headings, no bullet lists, no preamble like "Here is".
- Separate paragraphs with a blank line.

Source text:
{content}{language_reminder}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean(raw: str) -> str:
    """Strip any <think> block and surrounding whitespace."""
    return _THINK_RE.sub("", raw).strip()


def _normalize(text: str) -> str:
    """Lowercased, whitespace-collapsed form for duplicate comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


async def generate_digest(
    raw_content: str,
    *,
    summary: str | None = None,
    client: OllamaClient | None = None,
    keep_alive: str | int = 0,
) -> str | None:
    """Generate a long-form digest from raw_content, or None if not warranted.

    Gate + validation:
      * content shorter than digest.min_content_chars → None (no LLM call).
      * output shorter than digest.min_output_chars → None.
      * output nearly identical to ``summary`` → None.
      * any LLM error → None.

    ``client`` lets callers (batch/backfill) reuse a warm client; when None a
    fresh get_digest_llm_client() is created. ``keep_alive`` is passed to the
    query so batch callers can hold the model across items.
    """
    cfg = settings.user.digest
    content = (raw_content or "").strip()
    if len(content) < cfg.min_content_chars:
        return None

    if client is None:
        client = get_digest_llm_client()
    language = settings.user.slack.summary_language or "English"
    prompt = _DIGEST_PROMPT.format(
        content=content[: cfg.input_max_chars],
        language_reminder=language_directive(language),
    )
    try:
        raw = await client.query(
            "large",
            prompt,
            keep_alive=keep_alive,
            timeout=LARGE_MODEL_TIMEOUT,
            num_ctx=cfg.num_ctx,
            think=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_digest failed: %r", exc)
        return None

    digest = _clean(raw)
    if len(digest) < cfg.min_output_chars:
        return None
    if summary and _normalize(digest) == _normalize(summary):
        return None
    return digest


async def _digest_one(
    state: BrainState, client: OllamaClient, keep_alive: str | int
) -> BrainState:
    if not state.get("is_valid"):
        return {**state, "digest": None}
    digest = await generate_digest(
        state.get("raw_text") or "",
        summary=state.get("summary"),
        client=client,
        keep_alive=keep_alive,
    )
    return {**state, "digest": digest}


async def digest_node(state: BrainState) -> BrainState:
    """Single-item digest: load the digest model, generate, unload.

    Mirrors triage_node — keep_alive=0 then explicit unload so the 14B model
    does not linger in VRAM before the 32B genealogist may load.

    The min_content_chars gate is checked here too (ahead of client creation)
    so thin content makes zero HTTP calls at all — not even the unload ping —
    matching generate_digest's no-LLM-call guarantee.
    """
    if not state.get("is_valid"):
        return {**state, "digest": None}
    content = (state.get("raw_text") or "").strip()
    if len(content) < settings.user.digest.min_content_chars:
        return {**state, "digest": None}
    client = get_digest_llm_client()
    try:
        return await _digest_one(state, client, keep_alive=0)
    finally:
        try:
            await client.unload("large")
        except Exception:  # noqa: BLE001
            pass


async def batch_digest_states(
    states: list[BrainState],
    *,
    on_item_done: Callable[[], None] | None = None,
) -> list[BrainState]:
    """Digest all valid states with the 14B model loaded once (keep_alive=5m).

    Invalid states pass through with digest=None and no LLM call. The model is
    unloaded once after the batch (1 swap), matching batch_triage_states.
    """
    if not states:
        return []
    client = get_digest_llm_client()
    results: list[BrainState] = []
    try:
        for state in states:
            results.append(await _digest_one(state, client, keep_alive="5m"))
            if on_item_done is not None:
                try:
                    on_item_done()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("batch_digest_states on_item_done raised: %r", exc)
    finally:
        try:
            await client.unload("large")
        except Exception:  # noqa: BLE001
            pass
    return results
