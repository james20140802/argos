"""One-off backfill: clean HTML tags/entities from existing tech_items.title rows.

Usage:
    uv run python scripts/backfill_clean_titles.py [--dry-run]

Options:
    --dry-run   Print what would be changed without writing to the DB.

This script targets rows where the title contains a ``<`` or ``&`` character,
indicating potential HTML contamination.  It applies the same ``clean_title()``
function used by the crawler fetchers so the cleanup is consistent.

Note: Alembic migrations are locked (per project rules), so this script performs
the data fix directly via SQLAlchemy rather than through a migration.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import select, update

from argos.crawler._html_utils import clean_title
from argos.database import AsyncSessionLocal
from argos.models.tech_item import TechItem

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill(dry_run: bool = False) -> None:
    """Scan tech_items and clean any titles containing HTML artefacts."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TechItem.id, TechItem.title).where(
                TechItem.title.like("%<%") | TechItem.title.like("%&%")
            )
        )
        rows = result.all()

    if not rows:
        logger.info("No rows with potential HTML in title — nothing to do.")
        return

    logger.info("Found %d row(s) with potential HTML in title.", len(rows))

    updated = 0
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for row_id, raw_title in rows:
                cleaned = clean_title(raw_title)
                if cleaned == raw_title:
                    continue
                logger.info(
                    "  [%s] %r → %r",
                    row_id,
                    raw_title[:80],
                    cleaned[:80],
                )
                if not dry_run:
                    await session.execute(
                        update(TechItem)
                        .where(TechItem.id == row_id)
                        .values(title=cleaned)
                    )
                updated += 1

    if dry_run:
        logger.info("DRY RUN: would have updated %d row(s).", updated)
    else:
        logger.info("Updated %d row(s).", updated)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(backfill(dry_run=dry_run))
