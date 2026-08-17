from __future__ import annotations

import enum
import uuid

from sqlalchemy import Enum, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EntityKind(str, enum.Enum):
    """엔티티 종류. 분류 로직은 다음 사이클이므로 지금은 nullable로만 둔다 (A4)."""

    TECHNOLOGY = "Technology"
    ORGANIZATION = "Organization"
    PRODUCT = "Product"
    PERSON = "Person"
    OTHER = "Other"


class Entity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """사건에서 뽑힌 이름 하나 — 이름 사전(가제티어)의 한 row.

    ``name``은 화면에 보여줄 원문, ``normalized_key``는 같은 이름을 하나로
    모으기 위한 키다. unique 제약이 키에 걸려 있어 같은 이름이 두 row로
    쪼개지지 않는다. 정규화 *산식* 자체는 형제 이슈 ARG-225 소관이므로
    여기서는 최소 정규화(소문자화 + 공백 정리)까지만 한다 (A3).

    이 테이블이 남기는 목록은 다음 사이클 Technology/Organization 트리
    승인 큐의 후보가 된다 — 통째로 뽑아갈 수 있어야 한다.
    """

    __tablename__ = "entities"

    name: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
        index=True,
    )
    kind: Mapped[EntityKind | None] = mapped_column(
        Enum(
            EntityKind,
            name="entity_kind",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<Entity(name={self.name!r}, key={self.normalized_key!r})>"


class EventEntity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """사건↔엔티티 N:N 링크. 한 이름이 여러 사건에 걸쳐 재등장할 수 있다."""

    __tablename__ = "event_entities"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "entity_id",
            name="uq_event_entities_event_entity",
        ),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<EventEntity(event_id={self.event_id}, entity_id={self.entity_id})>"
