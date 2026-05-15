"""Async pgvector cosine similarity search for tech_items."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class SearchResult:
    title: str
    trust_score: Optional[float]
    category: Optional[str]
    status: Optional[str]
    created_at: datetime


async def search_tech_items(
    session: AsyncSession,
    embedding: list[float],
    limit: int = 10,
    category: Optional[str] = None,
    status: str = "all",
) -> list[SearchResult]:
    """Search tech_items by cosine similarity with optional category/status filters.

    Args:
        session: Async SQLAlchemy session.
        embedding: 768-dim query vector.
        limit: Max results (default 10, capped at 50).
        category: "alpha" | "mainstream" | None (no filter).
        status: "keep" (only Keep assets) | "all" (no filter, default).

    Returns:
        List of SearchResult ordered by cosine similarity descending.
    """
    limit = max(1, min(limit, 50))

    where_clauses = ["t.embedding IS NOT NULL"]
    params: dict = {
        "emb": "[" + ",".join(str(x) for x in embedding) + "]",
        "limit": limit,
    }

    if category is not None:
        where_clauses.append("t.category = :category")
        params["category"] = category.capitalize()

    if status == "keep":
        where_clauses.append("ua.status = 'Keep'")

    where_sql = " AND ".join(where_clauses)

    sql = text(
        f"SELECT t.title, t.trust_score, t.category, ua.status, t.created_at "
        f"FROM tech_items t "
        f"LEFT OUTER JOIN user_assets ua ON ua.tech_id = t.id "
        f"WHERE {where_sql} "
        f"ORDER BY t.embedding <=> CAST(:emb AS vector) "
        f"LIMIT :limit"
    )

    result = await session.execute(sql, params)
    rows = result.fetchall()

    return [
        SearchResult(
            title=row.title,
            trust_score=row.trust_score,
            category=row.category,
            status=row.status,
            created_at=row.created_at,
        )
        for row in rows
    ]
