from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AssignmentSource(str, enum.Enum):
    """배정을 누가 했나. 자동 배치는 ``Manual`` 행을 바꾸지 않는다."""

    AUTO = "Auto"
    # 신고 교정(ARG-317)이 쓸 자리.
    MANUAL = "Manual"


class RoleAssignment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """기술(대표 이름) -> 역할 아래 갈래 배정 한 행 — ARG-313.

    ``branch_id``가 NULL이면 **미분류**이고 ``unclassified_reason``에 사유가 있다.
    ``evidence``는 배정 근거(나열·대안 이웃, is-a 구, 외부 태그, 문장 수, LLM 이유)라
    ``argos taxonomy why``로 나중에 볼 수 있다.

    ``role_branches``는 읽기만 한다 — 갈래가 지워지면 CASCADE로 이 행만 사라지고
    다음 배치가 다시 배정한다. 갈래 확정본에 쓰는 경로는 여전히
    ``argos taxonomy apply`` 하나뿐이다.
    """

    __tablename__ = "role_assignments"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    branch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("role_branches.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    unclassified_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    source: Mapped[AssignmentSource] = mapped_column(
        Enum(
            AssignmentSource,
            name="assignment_source",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=AssignmentSource.AUTO,
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    def __repr__(self) -> str:
        return f"<RoleAssignment(entity_id={self.entity_id}, branch_id={self.branch_id})>"
