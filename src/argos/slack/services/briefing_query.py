from __future__ import annotations

from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus, UserAsset

KST = timezone(timedelta(hours=9))


async def fetch_today_briefing(
    session: AsyncSession,
    *,
    limit_per_category: int = 5,
    now_utc: datetime | None = None,
) -> dict[CategoryType, list[TechItem]]:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    now_kst = now_utc.astimezone(KST)
    start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    end_kst = start_kst + timedelta(days=1)

    start_utc = start_kst.astimezone(timezone.utc)
    end_utc = end_kst.astimezone(timezone.utc)

    result: dict[CategoryType, list[TechItem]] = {
        CategoryType.MAINSTREAM: [],
        CategoryType.ALPHA: [],
    }

    for category in (CategoryType.MAINSTREAM, CategoryType.ALPHA):
        stmt = (
            select(TechItem)
            .where(
                TechItem.category == category,
                TechItem.created_at >= start_utc,
                TechItem.created_at < end_utc,
            )
            .order_by(
                TechItem.trust_score.desc().nulls_last(),
                TechItem.created_at.desc(),
            )
            .limit(limit_per_category)
        )
        rows = await session.execute(stmt)
        result[category] = list(rows.scalars().all())

    return result


async def fetch_user_portfolio(
    session: AsyncSession,
) -> list[tuple[UserAsset, TechItem]]:
    stmt = (
        select(UserAsset, TechItem)
        .join(TechItem, UserAsset.tech_id == TechItem.id)
        .where(UserAsset.status == AssetStatus.KEEP)
        .order_by(UserAsset.updated_at.desc())
    )
    rows = await session.execute(stmt)
    return [(row[0], row[1]) for row in rows.all()]
