"""웹 푸시 구독과 아침 브리핑 발송 기록 (ARG-321).

* ``push_subscriptions`` — 알림을 켠 기기(브라우저) 하나당 한 행. 브라우저가
  준 ``endpoint``가 기기를 구분하는 키다(UNIQUE). 푸시 서비스가 404/410을
  돌려주면 만료로 보고 행을 지운다.
* ``push_dispatches`` — 아침 브리핑을 기기들에 보낸 회차마다 한 행(전부
  실패한 회차도 ``devices_ok = 0``으로 남는다). ``event_ids``는 그 회차에
  담긴 사건들 — 한 기기라도 받은 회차의 사건은 다음 회차에서 빠진다.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class PushSubscription(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """알림을 켠 기기 하나. 1인 사용자 — user_id 없음."""

    __tablename__ = "push_subscriptions"

    endpoint: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    p256dh: Mapped[str] = mapped_column(String(255), nullable=False)
    auth: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[str] = mapped_column(String(100), nullable=False, default="기기")
    # The https origin the device subscribed from (e.g. the tailnet address),
    # sent back to its push service as the VAPID ``sub`` contact.
    origin: Mapped[str] = mapped_column(String(255), nullable=False)
    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<PushSubscription(id={self.id}, label={self.label!r})>"


class PushDispatch(UUIDPrimaryKeyMixin, Base):
    """아침 브리핑 발송 한 회차 (실패한 회차 포함, ``devices_ok``로 구분)."""

    __tablename__ = "push_dispatches"

    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    event_count: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    devices_ok: Mapped[int] = mapped_column(Integer, nullable=False)
    devices_failed: Mapped[int] = mapped_column(Integer, nullable=False)
    devices_expired: Mapped[int] = mapped_column(Integer, nullable=False)
    # Events this dispatch carried. Dedup is by id, not by ``sent_at``: an
    # event committed after a dispatch can carry an earlier ``created_at``
    # (the crawl transaction's start), so a time watermark would skip it.
    event_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, server_default=text("'{}'")
    )

    def __repr__(self) -> str:
        return f"<PushDispatch(sent_at={self.sent_at}, events={self.event_count})>"
