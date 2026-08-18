from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TechEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """사건 한 건 — 여러 문서(기사)가 근거로 매달리는 단위.

    ``feed_events``(ARG-207의 Impression/Click/Dwell 행동 로그)와는 완전히
    다른 개념이다. 이름이 헷갈리지 않도록 테이블명을 ``tech_events``로 못박는다.

    병합은 삭제가 아니라 **툼스톤**이다. 사건 A가 사건 B에 흡수될 때 A의 row는
    ``DELETE`` 하지 않고 ``merged_into_id = B.id``만 채운다. 그래야 A의 id를 담은
    옛 링크가 죽지 않고 B로 이어진다. self-FK에 ``ondelete="RESTRICT"``를 건 것도
    같은 이유 — 누군가 흡수해 간 사건은 DB 레벨에서 삭제를 거부한다.

    단, FK만 걸어서는 그 방어가 서지 않는다. ``AsyncSession``으로 B를 지우면
    SQLAlchemy의 기본 delete synchronization이 A의 ``merged_into_id``를 먼저
    ``NULL``로 밀어버리고, 그러면 Postgres는 참조하는 행을 못 보므로 RESTRICT가
    발동하지 않는다 — 삭제는 조용히 성공하고 A의 id는 더 이상 B로 이어지지
    않는다. 아래 ``merged_from``의 ``passive_deletes="all"``이 그 사전 NULL
    처리를 끄고 FK가 실제로 말하게 한다. 회귀 테스트:
    ``tests/test_tech_event_tombstone_db.py``.
    """

    __tablename__ = "tech_events"

    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    naming_stale: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_events.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    merged_into = relationship(
        "TechEvent",
        remote_side=lambda: [TechEvent.id],
        back_populates="merged_from",
    )
    merged_from = relationship(
        "TechEvent",
        back_populates="merged_into",
        # 이미 세션에 로드된 자식까지 포함해 ORM이 FK를 건드리지 않게 한다.
        # 이게 없으면 위 docstring대로 RESTRICT가 우회된다.
        passive_deletes="all",
    )

    def __repr__(self) -> str:
        return (
            f"<TechEvent(id={self.id}, title={self.title!r}, "
            f"merged_into_id={self.merged_into_id})>"
        )
