from __future__ import annotations

import enum
import uuid
from typing import Any

from sqlalchemy import CheckConstraint, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AliasRule(str, enum.Enum):
    """이름 하나가 다른 이름의 별칭이 된 근거 규칙."""

    SPACING = "Spacing"
    VENDOR_PREFIX = "VendorPrefix"
    # 신고 교정(ARG-317)이 쓸 자리. 자동 재계산은 이 행을 지우거나 바꾸지 않는다.
    MANUAL = "Manual"


class EntityAlias(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """이름 → 대표 이름 별칭 링크 — ARG-312.

    ``entities``는 건드리지 않고 그 위에 얹는 층이다. 저장된 이름(``normalized_key``)과
    ``document_entities``는 사건 묶기가 읽는 입력이라 그대로 둔다.

    한 이름은 별칭 행을 최대 하나만 가진다(``alias_entity_id`` UNIQUE). 대표 이름은
    별칭 행이 없는 쪽이고, 사슬은 저장하지 않는다 — 쓰는 쪽이 대표 하나로 펼쳐서
    넣는다. CASCADE는 **별칭 행만** 지운다. ``entities`` 행은 지우지 않는다.
    """

    __tablename__ = "entity_aliases"
    __table_args__ = (
        CheckConstraint(
            "alias_entity_id <> canonical_entity_id",
            name="ck_entity_aliases_not_self",
        ),
    )

    alias_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    canonical_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rule: Mapped[AliasRule] = mapped_column(
        Enum(
            AliasRule,
            name="alias_rule",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    # 판정 근거(접두어 등장 문서 수·소문자 비율 등). 화면에 쓰지 않고 검토용이다.
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    def __repr__(self) -> str:
        return (
            f"<EntityAlias(alias={self.alias_entity_id}, "
            f"canonical={self.canonical_entity_id}, rule={self.rule})>"
        )
