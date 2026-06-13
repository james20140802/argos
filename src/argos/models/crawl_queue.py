from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base


class CrawlQueue(Base):
    """Staging table that buffers crawled items before brain-pipeline processing.

    Items enter via upsert-on-conflict-do-nothing (source_url is UNIQUE) and are
    deleted once the brain pipeline has processed them.  Unprocessed items survive
    across runs and are re-prioritised by published_at on the next run.
    """

    __tablename__ = "crawl_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    image_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
