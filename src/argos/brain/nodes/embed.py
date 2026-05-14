from __future__ import annotations
import logging
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

        embedding = await get_embedding(state["raw_text"][:3000])
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        stmt = text(
            "SELECT id, title, raw_content FROM tech_items "
            "WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> CAST(:emb AS vector) LIMIT 5"
        )
        result = await session.execute(stmt, {"emb": embedding_str})
        rows = result.fetchall()
        related_ids = [str(r.id) for r in rows]
        similar_items = [
            {"id": str(r.id), "title": r.title, "raw_content": r.raw_content[:500]}
            for r in rows
        ]
        # Defensive: if the Top-5 query returns nothing despite the count check,
        # treat this run as a cold start too — genealogist has nothing to compare.
        if not rows:
            logger.info(
                "Top-5 similarity search returned no rows despite %d embedded items; "
                "skipping genealogy",
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


async def batch_embed_and_search_node(
    states: list[BrainState],
    session: AsyncSession,
) -> list[BrainState]:
    """Embed all valid states in a single /api/embed call, then run similarity search per item.

    Returns a parallel list of states; invalid states are passed through unchanged.
    """
    from argos.config import settings as _settings

    top_n = _settings.user.genealogist.context_top_n
    max_chars = _settings.user.genealogist.context_max_chars
    threshold = _settings.user.genealogist.min_db_items

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

    out = list(states)
    for pos, idx in enumerate(valid_indices):
        state = states[idx]
        embedding = embeddings[pos]

        if cold_start:
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
            continue

        try:
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            stmt = text(
                f"SELECT id, title, raw_content FROM tech_items "
                f"WHERE embedding IS NOT NULL "
                f"ORDER BY embedding <=> CAST(:emb AS vector) LIMIT {top_n}"
            )
            result = await session.execute(stmt, {"emb": embedding_str})
            rows = result.fetchall()
        except Exception as exc:
            logger.warning("batch_embed_and_search_node: similarity search failed: %r", exc)
            out[idx] = {**state, "related_tech_ids": [], "extracted_info": None}
            continue

        if not rows:
            out[idx] = {
                **state,
                "related_tech_ids": [],
                "extracted_info": {"embedding": embedding, "similar_items": []},
                "genealogy_skipped": True,
                "genealogy_skip_reason": "cold_start",
            }
        else:
            out[idx] = {
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
    return out
