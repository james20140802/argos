"""Read-side service backing the 관측 피드 screen (ARG-155).

``fetch_feed`` returns recent ``tech_items`` joined to the user's asset
status. The service deliberately ignores the column exclusively owned by
the Slack briefing pipeline so the two surfaces stay decoupled.
"""
from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus, UserAsset


PAGE_SIZE: int = 20


Category = Literal["Mainstream", "Alpha"]


@dataclass(frozen=True)
class FeedItem:
    id: uuid.UUID
    title: str
    source_url: str
    category: Optional[CategoryType]
    image_url: Optional[str]
    status: Optional[AssetStatus]
    sort_at: datetime


@dataclass(frozen=True)
class FeedPage:
    items: list[FeedItem]
    next_cursor: Optional[str]


# ------------------------------------------------------------------ #
# Cursor helpers
# ------------------------------------------------------------------ #

def encode_cursor(sort_at: datetime, item_id: uuid.UUID) -> str:
    if sort_at.tzinfo is None:
        sort_at = sort_at.replace(tzinfo=timezone.utc)
    payload = {"t": sort_at.astimezone(timezone.utc).isoformat(), "i": item_id.hex}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(token: str) -> tuple[datetime, uuid.UUID]:
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        sort_at = datetime.fromisoformat(payload["t"])
        if sort_at.tzinfo is None:
            sort_at = sort_at.replace(tzinfo=timezone.utc)
        item_id = uuid.UUID(payload["i"])
        return sort_at, item_id
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feed cursor: {token!r}") from exc


# ------------------------------------------------------------------ #
# Query
# ------------------------------------------------------------------ #

async def fetch_feed(
    session: AsyncSession,
    *,
    category: Optional[Category] = None,
    cursor: Optional[str] = None,
    limit: int = PAGE_SIZE,
) -> FeedPage:
    """Return one paginated page of feed items.

    The Slack briefing column is intentionally not referenced here —
    that column is exclusively owned by the briefing pipeline.
    """
    sort_expr = func.coalesce(TechItem.published_at, TechItem.created_at)

    stmt = (
        select(
            TechItem.id,
            TechItem.title,
            TechItem.source_url,
            TechItem.category,
            TechItem.image_url,
            UserAsset.status,
            sort_expr.label("sort_at"),
        )
        .join(UserAsset, UserAsset.tech_id == TechItem.id, isouter=True)
        .order_by(sort_expr.desc(), TechItem.id.desc())
        .limit(limit + 1)
    )

    if category is not None:
        if category not in ("Mainstream", "Alpha"):
            raise ValueError(f"invalid category: {category!r}")
        stmt = stmt.where(TechItem.category == CategoryType(category))

    if cursor is not None:
        cur_sort, cur_id = decode_cursor(cursor)
        stmt = stmt.where(
            (sort_expr < cur_sort)
            | ((sort_expr == cur_sort) & (TechItem.id < cur_id))
        )

    result = await session.execute(stmt)
    rows = result.all()

    items = [
        FeedItem(
            id=row.id,
            title=row.title,
            source_url=row.source_url,
            category=row.category,
            image_url=row.image_url,
            status=row.status,
            sort_at=row.sort_at,
        )
        for row in rows[:limit]
    ]

    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = encode_cursor(last.sort_at, last.id)

    return FeedPage(items=items, next_cursor=next_cursor)
