from __future__ import annotations
import logging
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.graph_state import BrainState
from argos.brain.ollama_client import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

async def get_embedding(text_input: str) -> list[float]:
    payload = {"model": "nomic-embed-text", "prompt": text_input}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()["embedding"]

async def embed_and_search_node(state: BrainState, session: AsyncSession) -> BrainState:
    if not state["is_valid"]:
        return state
    try:
        embedding = await get_embedding(state["raw_text"][:3000])
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        stmt = text(
            "SELECT id, title, raw_content FROM tech_items "
            "ORDER BY embedding <=> CAST(:emb AS vector) LIMIT 5"
        )
        result = await session.execute(stmt, {"emb": embedding_str})
        rows = result.fetchall()
        related_ids = [str(r.id) for r in rows]
        similar_items = [
            {"id": str(r.id), "title": r.title, "raw_content": r.raw_content[:500]}
            for r in rows
        ]
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
