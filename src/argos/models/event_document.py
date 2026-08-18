from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EventDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """사건↔문서 N:N 근거 링크.

    ``tech_items``에 ``event_id`` 컬럼을 다는 대신 별도 링크 테이블을 쓴다 (A2).
    그래야 기존 문서(코퍼스 전체)의 스키마가 무변경으로 남고, 한 문서가 두
    사건의 근거가 되는 경우도 열린다.
    """

    __tablename__ = "event_documents"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "tech_item_id",
            name="uq_event_documents_event_item",
        ),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tech_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<EventDocument(event_id={self.event_id}, "
            f"tech_item_id={self.tech_item_id})>"
        )
