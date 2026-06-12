from __future__ import annotations
import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node, batch_triage_states
from argos.brain.nodes.embed import embed_and_search_node, batch_embed_and_search_node
from argos.brain.nodes.genealogist import genealogist_node
from argos.brain.nodes.save import save_node
from argos.brain.llm_client import get_genealogist_llm_client
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)


async def run_brain_pipeline(
    raw_text: str,
    source_url: str,
    session: AsyncSession,
    *,
    source_category: CategoryType | None = None,
    published_at: datetime | None = None,
    image_url: str | None = None,
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
        "published_at": published_at,
        "image_url": image_url,
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

    trust_score = triaged.get("trust_score")
    from argos.config import settings as _settings
    threshold = _settings.user.genealogist.trust_skip_threshold
    if trust_score is not None and trust_score < threshold:
        skipped: BrainState = {
            **embedded,
            "genealogy_skipped": True,
            "genealogy_skip_reason": "low_trust",
        }
        return await save_node(skipped, session=session)

    prewarm_task = asyncio.create_task(get_genealogist_llm_client().prewarm("large"))
    try:
        genealogized = await genealogist_node(embedded, prewarm_task=prewarm_task)
        return await save_node(genealogized, session=session)
    finally:
        if not prewarm_task.done():
            prewarm_task.cancel()
        with contextlib.suppress(BaseException):
            await prewarm_task


def _make_initial_state(item: dict) -> BrainState:
    source_category = item.get("_source_category")
    return {
        "raw_text": item.get("raw_content") or "",
        "source_url": item.get("source_url", "").strip(),
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
        "published_at": item.get("_published_at"),
        "image_url": item.get("image_url"),
    }


async def run_batch_brain_pipeline(
    items: list[dict],
    session: AsyncSession,
    *,
    on_triage_item_done: Callable[[], None] | None = None,
    on_embed_item_done: Callable[[], None] | None = None,
    on_genealogy_item_done: Callable[[], None] | None = None,
    on_save_item_done: Callable[[], None] | None = None,
) -> list[BrainState]:
    """Process N items through the brain pipeline with 3 model swaps total.

    Stage order: batch triage (8B × 1 swap) → batch embed (/api/embed × 1 call)
    → trust-score gate → batch genealogy (32B × 1 swap) → per-item save.

    The single-URL run_brain_pipeline is preserved for backwards compatibility
    and single-URL callers (e.g. tests, future Slack Deep-Dive paths).

    Parameters
    ----------
    items, session:
        See module-level usage.
    on_triage_item_done, on_embed_item_done, on_genealogy_item_done,
    on_save_item_done:
        Optional zero-arg callbacks for per-item progress reporting in each
        stage. ``on_triage_item_done`` / ``on_embed_item_done`` are forwarded
        to ``batch_triage_states`` / ``batch_embed_and_search_node``;
        ``on_genealogy_item_done`` fires inside the genealogy loop once per
        candidate; ``on_save_item_done`` fires once per state in the save
        loop (including invalid states that are skipped, so the bar reflects
        every queue slot). Default ``None`` preserves existing behavior —
        they are the UI-injection point used by the CLI to drive a Rich
        progress bar (ARG-92/ARG-101). Exceptions raised by callbacks are
        swallowed so a broken UI cannot abort the pipeline.
    """
    from argos.config import settings as _settings

    if not items:
        return []

    states = [_make_initial_state(item) for item in items]

    # ── Stage 1: batch triage (8B loaded once) ────────────────────────────
    triaged_states = await batch_triage_states(
        states, on_item_done=on_triage_item_done
    )

    # ── Stage 2: batch embed + similarity search ──────────────────────────
    embedded_states = await batch_embed_and_search_node(
        triaged_states, session, on_item_done=on_embed_item_done
    )

    # ── Stage 3: trust-score gate + batch genealogy (32B loaded once) ─────
    threshold = _settings.user.genealogist.trust_skip_threshold
    genealogy_candidates: list[int] = []
    for i, s in enumerate(embedded_states):
        if not s.get("is_valid"):
            continue
        if s.get("genealogy_skipped"):
            continue
        trust = s.get("trust_score")
        if trust is not None and trust < threshold:
            embedded_states[i] = {
                **s,
                "genealogy_skipped": True,
                "genealogy_skip_reason": "low_trust",
            }
            continue
        genealogy_candidates.append(i)

    if genealogy_candidates:
        prewarm_task = asyncio.create_task(get_genealogist_llm_client().prewarm("large"))
        try:
            passed_prewarm = False
            for i in genealogy_candidates:
                try:
                    embedded_states[i] = await genealogist_node(
                        embedded_states[i],
                        prewarm_task=prewarm_task if not passed_prewarm else None,
                    )
                    passed_prewarm = True
                finally:
                    if on_genealogy_item_done is not None:
                        try:
                            on_genealogy_item_done()
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "run_batch_brain_pipeline on_genealogy_item_done raised: %r",
                                exc,
                            )
        finally:
            if not prewarm_task.done():
                prewarm_task.cancel()
            with contextlib.suppress(BaseException):
                await prewarm_task
        # Unload 32B after all genealogy work is done.
        try:
            await get_genealogist_llm_client().unload("large")
        except Exception:
            pass

    # ── Stage 4: per-item save (savepoint + flush per item) ──────────────────
    #
    # Each item is saved inside a begin_nested() savepoint and flushed within
    # that savepoint.  Flushing inside the savepoint ensures that DB-deferred
    # constraint violations (unique, FK, vector) are caught by the surrounding
    # except block and mark only that item as failed, rather than aborting the
    # entire batch.  The flush=False flag on save_node means save_node itself
    # does not issue the flush; the savepoint block does it explicitly after
    # save_node returns, giving us the same pre-flush logic-error isolation
    # while also catching post-flush constraint errors per item.
    results: list[BrainState] = []
    for s in embedded_states:
        try:
            if not s.get("source_url"):
                logger.warning("run_batch_brain_pipeline: state missing source_url, skipping")
                results.append(s)
                continue
            try:
                async with session.begin_nested():
                    saved = await save_node(s, session=session, flush=False)
                    await session.flush()
                    saved["saved"] = True
                results.append(saved)
            except Exception as exc:
                logger.warning(
                    "run_batch_brain_pipeline: save failed for %s: %r",
                    s.get("source_url"),
                    exc,
                )
                results.append(s)
        finally:
            if on_save_item_done is not None:
                try:
                    on_save_item_done()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "run_batch_brain_pipeline on_save_item_done raised: %r",
                        exc,
                    )
    return results
