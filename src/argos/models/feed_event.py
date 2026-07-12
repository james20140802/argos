import enum
import uuid

from sqlalchemy import Enum, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class FeedEventType(str, enum.Enum):
    IMPRESSION = "Impression"
    CLICK = "Click"
    DWELL = "Dwell"


class FeedEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """피드 상호작용 이벤트(학습 랭커용 원재료). 1인 사용자 — user_id 없음."""

    __tablename__ = "feed_events"

    event_type: Mapped[FeedEventType] = mapped_column(
        Enum(
            FeedEventType,
            name="feed_event_type",
            values_callable=lambda enum_cls: [m.value for m in enum_cls],
        ),
        nullable=False,
    )
    tech_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_items.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    value: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<FeedEvent(id={self.id}, event_type={self.event_type})>"
