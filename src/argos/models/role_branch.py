from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RoleBranch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """확정된 역할 갈래 한 행 — ARG-311.

    기술이 "무슨 역할인가"를 묻는 분류 틀(역할 축)의 **확정본**이다. 행이
    존재한다는 것 자체가 사람이 확정했다는 뜻이고, 초안은 DB에 두지 않는다
    (JSON 파일). 위 갈래는 ``parent_id IS NULL``, 아래 갈래는 위 갈래를
    가리킨다(깊이 2단은 ``argos.brain.taxonomy.validate``가 지킨다).

    이 테이블을 쓰는 경로는 ``argos taxonomy apply`` 하나뿐이다 — 크롤·교정·
    스케줄 같은 자동 처리는 읽기만 한다 (``tests/brain/test_taxonomy_isolation.py``).
    이름이 전역 유일이라 명령에서 ``--branch "코딩 에이전트"``처럼 가리킬 수 있다.

    계열 축(제품 계통)은 3단계에서 별도 테이블로 둔다 — 두 축은 다른 트리다.
    """

    __tablename__ = "role_branches"

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("role_branches.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<RoleBranch(name={self.name!r}, parent_id={self.parent_id})>"
