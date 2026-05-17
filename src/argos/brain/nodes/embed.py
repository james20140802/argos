from __future__ import annotations
import asyncio
import logging
from typing import Callable
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.graph_state import BrainState
from argos.brain.ollama_client import _base_url, batch_embed
from argos.config import settings

logger = logging.getLogger(__name__)

async def get_embedding(text_input: str) -> list[float]:
    payload = {"model": "nomic-embed-text", "prompt": text_input, "keep_alive": 0}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{_base_url()}/api/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()["embedding"]

async def embed_and_search_node(state: BrainState, session: AsyncSession) -> BrainState:
    if not state["is_valid"]:
        return state
    try:
        # Cold-start guard: count embedded items and short-circuit the 32B
        # genealogist branch when the corpus is below the configured threshold.
        threshold = settings.user.genealogist.min_db_items
        count_stmt = text(
            "SELECT count(*) AS n FROM tech_items WHERE embedding IS NOT NULL"
        )
        count_result = await session.execute(count_stmt)
        embedded_count = int(count_result.scalar() or 0)
        if embedded_count < threshold:
            logger.info(
                "DB items insufficient for genealogy (%d items, threshold=%d), skipping",
                embedded_count,
                threshold,
            )
            embedding = await get_embedding(state["raw_text"][:3000])
            return {
                **state,
                "related_tech_ids": [],
                "extracted_info": {
                    "embedding": embedding,
                    "similar_items": [],
                },
                "genealogy_skipped": True,
                "genealogy_skip_reason": "cold_start",
            }

        top_n = settings.user.genealogist.context_top_n
        max_chars = settings.user.genealogist.context_max_chars

        embedding = await get_embedding(state["raw_text"][:3000])
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        stmt = text(
            f"SELECT id, title, raw_content FROM tech_items "
            f"WHERE embedding IS NOT NULL "
            f"ORDER BY embedding <=> CAST(:emb AS vector) LIMIT {top_n}"
        )
        result = await session.execute(stmt, {"emb": embedding_str})
        rows = result.fetchall()
        related_ids = [str(r.id) for r in rows]
        similar_items = [
            {"id": str(r.id), "title": r.title, "raw_content": r.raw_content[:max_chars]}
            for r in rows
        ]
        if not rows:
            logger.info(
                "Top-%d similarity search returned no rows despite %d embedded items; "
                "skipping genealogy",
                top_n,
                embedded_count,
            )
            return {
                **state,
                "related_tech_ids": [],
                "extracted_info": {
                    "embedding": embedding,
                    "similar_items": [],
                },
                "genealogy_skipped": True,
                "genealogy_skip_reason": "cold_start",
            }
        return {
            **state,
            "related_tech_ids": related_ids,
            "extracted_info": {
                "embedding": embedding,
                "similar_items": similar_items,
            },
        }
    except Exception as exc:
        logger.warning("embed_and_search_node failed: %r", exc)
        return {**state, "related_tech_ids": [], "extracted_info": None}


async def _similarity_search(
    state: BrainState,
    embedding: list[float],
    session: AsyncSession,
    top_n: int,
    max_chars: int,
) -> BrainState:
    """Run a single pgvector similarity search for *state* using *embedding*.

    Returns an updated copy of *state* with ``related_tech_ids`` and
    ``extracted_info`` populated, or with ``genealogy_skipped=True`` when the
    search returns no rows.

    This helper is intentionally kept pure (no side-effects beyond the DB
    read) so it can be called concurrently from ``batch_embed_and_search_node``
    via ``asyncio.gather`` + ``asyncio.Semaphore``.
    """
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    stmt = text(
        f"SELECT id, title, raw_content FROM tech_items "
        f"WHERE embedding IS NOT NULL "
        f"ORDER BY embedding <=> CAST(:emb AS vector) LIMIT {top_n}"
    )
    result = await session.execute(stmt, {"emb": embedding_str})
    rows = result.fetchall()

    if not rows:
        return {
            **state,
            "related_tech_ids": [],
            "extracted_info": {"embedding": embedding, "similar_items": []},
            "genealogy_skipped": True,
            "genealogy_skip_reason": "cold_start",
        }
    return {
        **state,
        "related_tech_ids": [str(r.id) for r in rows],
        "extracted_info": {
            "embedding": embedding,
            "similar_items": [
                {
                    "id": str(r.id),
                    "title": r.title,
                    "raw_content": r.raw_content[:max_chars],
                }
                for r in rows
            ],
        },
    }


async def batch_embed_and_search_node(
    states: list[BrainState],
    session: AsyncSession,
    *,
    on_item_done: Callable[[], None] | None = None,
) -> list[BrainState]:
    """Embed all valid states in a single /api/embed call, then run similarity
    searches concurrently (bounded by ``genealogist.embed_search_concurrency``).

    Returns a parallel list of states; invalid states are passed through unchanged.

    Parameters
    ----------
    states:
        Input batch of brain states.
    session:
        Active async SQLAlchemy session.  Concurrent coroutines share this
        session; SQLAlchemy's asyncio implementation serialises access to the
        underlying connection internally, so no session-safety regression is
        introduced.  The semaphore bounds the fan-out to
        ``settings.user.genealogist.embed_search_concurrency`` (default 4) so
        the asyncpg pool is never overwhelmed.
    on_item_done:
        Optional zero-arg callback invoked once per *valid* state after its
        embedding + similarity search slot completes (i.e., the number of
        ticks equals ``len([s for s in states if s['is_valid']])``).
        Invalid states are passed through with no work and do not tick.
        Defaults to ``None`` (no-op). Exceptions in the callback are
        swallowed so UI failures cannot abort the pipeline.
    """
    from argos.config import settings as _settings

    top_n = _settings.user.genealogist.context_top_n
    max_chars = _settings.user.genealogist.context_max_chars
    threshold = _settings.user.genealogist.min_db_items
    concurrency = _settings.user.genealogist.embed_search_concurrency

    valid_indices = [i for i, s in enumerate(states) if s.get("is_valid")]
    if not valid_indices:
        return list(states)

    try:
        count_stmt = text(
            "SELECT count(*) AS n FROM tech_items WHERE embedding IS NOT NULL"
        )
        count_result = await session.execute(count_stmt)
        embedded_count = int(count_result.scalar() or 0)
        cold_start = embedded_count < threshold
    except Exception as exc:
        logger.warning("batch_embed_and_search_node: DB count failed: %r", exc)
        return list(states)

    texts = [states[i]["raw_text"][:3000] for i in valid_indices]
    try:
        embeddings = await batch_embed(texts)
    except Exception as exc:
        logger.warning("batch_embed_and_search_node: batch_embed failed: %r", exc)
        return list(states)

    def _tick() -> None:
        if on_item_done is None:
            return
        try:
            on_item_done()
        except Exception as exc:  # noqa: BLE001
            logger.debug("batch_embed_and_search_node on_item_done raised: %r", exc)

    out = list(states)

    if cold_start:
        # Cold-start path: no similarity searches needed; populate embeddings only.
        for pos, idx in enumerate(valid_indices):
            state = states[idx]
            embedding = embeddings[pos]
            logger.info(
                "DB items insufficient for genealogy (%d items, threshold=%d), skipping",
                embedded_count,
                threshold,
            )
            out[idx] = {
                **state,
                "related_tech_ids": [],
                "extracted_info": {"embedding": embedding, "similar_items": []},
                "genealogy_skipped": True,
                "genealogy_skip_reason": "cold_start",
            }
            _tick()
        return out

    # Warm path: run similarity searches concurrently, bounded by semaphore.
    sem = asyncio.Semaphore(concurrency)

    async def _search_one(pos: int, idx: int) -> tuple[int, BrainState]:
        """Run one similarity search and return (idx, updated_state)."""
        state = states[idx]
        embedding = embeddings[pos]
        try:
            async with sem:
                updated = await _similarity_search(
                    state, embedding, session, top_n, max_chars
                )
        except Exception as exc:
            logger.warning(
                "batch_embed_and_search_node: similarity search failed: %r", exc
            )
            updated = {**state, "related_tech_ids": [], "extracted_info": None}
        finally:
            _tick()
        return idx, updated

    results = await asyncio.gather(
        *(_search_one(pos, idx) for pos, idx in enumerate(valid_indices))
    )
    for idx, updated_state in results:
        out[idx] = updated_state

    return out
