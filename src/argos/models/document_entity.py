from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DocumentEntity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """문서↔엔티티 N:N 링크 — ARG-262.

    이름을 **사건이 아니라 문서에** 매단다. 사건 쪽 링크(``event_entities``)만
    있으면 사건이 하나도 없는 초기에 이웃의 이름이 항상 비어 간선 가중치의
    이름 항(0.25)이 통째로 죽고, 그러면 초반 전체가 임베딩만으로 묶인다.
    """

    __tablename__ = "document_entities"
    __table_args__ = (
        UniqueConstraint(
            "tech_item_id",
            "entity_id",
            name="uq_document_entities_item_entity",
        ),
    )

    # index=True를 걸지 않는다 — uq_document_entities_item_entity의 선행
    # 컬럼이 tech_item_id라 단독 조회도 그 인덱스로 처리된다. 따로 만들면
    # 쓰기 비용만 늘어난다 (event_documents와 같은 판단).
    tech_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentEntity(tech_item_id={self.tech_item_id}, "
            f"entity_id={self.entity_id})>"
        )
