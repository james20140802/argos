import enum

from sqlalchemy import Enum, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CategoryType(str, enum.Enum):
    """기술 카테고리: 대세(Mainstream) vs 혁신(Alpha)."""

    MAINSTREAM = "Mainstream"
    ALPHA = "Alpha"


class TechItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """수집된 기술 정보를 저장하는 핵심 테이블."""

    __tablename__ = "tech_items"

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(1536), nullable=True)
    category: Mapped[CategoryType] = mapped_column(
        Enum(CategoryType, name="category_type"),
        nullable=True,
    )
    trust_score: Mapped[float] = mapped_column(Float, nullable=True, default=0.0)

    # Relationships
    predecessors = relationship(
        "TechSuccession",
        foreign_keys="TechSuccession.successor_id",
        back_populates="successor",
        lazy="selectin",
    )
    successors = relationship(
        "TechSuccession",
        foreign_keys="TechSuccession.predecessor_id",
        back_populates="predecessor",
        lazy="selectin",
    )
    user_assets = relationship(
        "UserAsset",
        back_populates="tech_item",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<TechItem(id={self.id}, title='{self.title[:30]}...')>"
