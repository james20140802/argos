import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from argos.models.base import Base, UUIDPrimaryKeyMixin


class TrackHistory(UUIDPrimaryKeyMixin, Base):
    """user_assets의 상태 변경 이력을 기록하는 로그 테이블."""

    __tablename__ = "track_history"

    user_asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user_assets.id", ondelete="CASCADE"),
        nullable=False,
    )
    changed_from: Mapped[str] = mapped_column(String(50), nullable=False)
    changed_to: Mapped[str] = mapped_column(String(50), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user_asset = relationship("UserAsset", back_populates="history")

    def __repr__(self) -> str:
        return (
            f"<TrackHistory({self.changed_from} -> {self.changed_to} "
            f"at {self.changed_at})>"
        )
