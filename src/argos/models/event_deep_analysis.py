"""사건별 심층 분석 저장 자리 (ARG-333).

사건 하나당 한 행이고, 한 행 안에 서로 독립인 두 묶음이 있다.

* **최근 성공 결과** — ``sections`` · ``analyzed_at`` · ``doc_count`` · ``doc_ids`` ·
  ``model``. 다시 분석이 성공할 때만 덮어쓴다(이력 없음).
* **지금 진행 상태** — ``status`` · ``requested_at`` · ``started_at`` ·
  ``failure_reason``. ``status``가 NULL이면 진행 중인 것도 실패한 것도 없다.

둘을 한 묶음으로 합치지 않는 이유: 다시 분석이 실패해도 직전 성공 결과는
화면에 남아야 한다. 그래서 진행·실패 칸을 바꿔도 결과 칸은 그대로다.

진행 상태를 웹 프로세스 메모리가 아니라 여기 두는 이유: 웹이 재시작되면
메모리의 흔적이 사라져, 끊긴 분석을 "실패"로 보여 줄 수 없다.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DeepAnalysisStatus(str, enum.Enum):
    """진행 상태. 컬럼이 NULL이면 "없음"(대기도 진행도 실패도 아님)."""

    WAITING = "Waiting"
    """다른 작업이 모델을 쓰는 중이라 순번을 기다린다."""
    RUNNING = "Running"
    FAILED = "Failed"


class EventDeepAnalysis(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """사건 하나의 심층 분석 — 최근 성공 결과 하나 + 지금 진행 상태 하나."""

    __tablename__ = "event_deep_analyses"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tech_events.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # ── 최근 성공 결과 (한 번도 성공한 적 없으면 전부 NULL) ──
    # ``[{"heading": ..., "body": ...}]`` 다섯 개 —
    # ``brain.event_deep_analysis.DeepAnalysis.sections_json()``과 같은 모양.
    sections: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    doc_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 분석 당시 기사 묶음. 건수만 두면 1건 빠지고 1건 붙은 경우를 놓친다.
    doc_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # ── 지금 진행 상태 (결과 칸과 독립으로 비운다) ──
    status: Mapped[DeepAnalysisStatus | None] = mapped_column(
        Enum(
            DeepAnalysisStatus,
            name="deep_analysis_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=True,
    )
    requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    failure_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)

    def __repr__(self) -> str:
        return f"<EventDeepAnalysis(event_id={self.event_id}, status={self.status})>"
