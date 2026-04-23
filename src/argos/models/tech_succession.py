import enum
import uuid

from sqlalchemy import Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RelationType(str, enum.Enum):
    """기술 계승 관계 유형."""

    REPLACE = "Replace"
    ENHANCE = "Enhance"
    FORK = "Fork"


class TechSuccession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """기술 가계도 — 기술 간 계승 관계를 저장한다."""

    __tablename__ = "tech_succession"

    predecessor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    successor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[RelationType] = mapped_column(
        Enum(
            RelationType,
            name="relation_type",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    reasoning: Mapped[str] = mapped_column(Text, nullable=True)

    # Relationships
    predecessor = relationship(
        "TechItem",
        foreign_keys=[predecessor_id],
        back_populates="successors",
    )
    successor = relationship(
        "TechItem",
        foreign_keys=[successor_id],
        back_populates="predecessors",
    )

    def __repr__(self) -> str:
        return (
            f"<TechSuccession({self.predecessor_id} "
            f"--[{self.relation_type.value}]--> {self.successor_id})>"
        )
