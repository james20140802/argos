from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AssetStatus(str, enum.Enum):
    """사용자 자산 상태."""

    KEEP = "Keep"
    TRACKING = "Tracking"
    ARCHIVED = "Archived"


class UserAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """사용자가 Keep한 기술 자산을 관리한다."""

    __tablename__ = "user_assets"

    tech_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[AssetStatus] = mapped_column(
        Enum(AssetStatus, name="asset_status"),
        nullable=False,
        default=AssetStatus.KEEP,
    )
    last_monitored_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    tech_item = relationship("TechItem", back_populates="user_assets")
    history = relationship(
        "TrackHistory",
        back_populates="user_asset",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<UserAsset(id={self.id}, status={self.status.value})>"
