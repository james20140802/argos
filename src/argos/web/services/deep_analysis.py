"""사건 심층 분석 — 웹이 읽고 쓰는 서비스 (ARG-336).

저장 모양은 ``models.event_deep_analysis``: 사건당 한 행에 "최근 성공 결과"와
"지금 진행 상태"가 독립으로 들어 있다. 여기 함수들은 둘을 따로 만진다.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from typing import TYPE_CHECKING

from argos.models.event_deep_analysis import DeepAnalysisStatus, EventDeepAnalysis

if TYPE_CHECKING:  # argos.brain 패키지는 argos.database를 끌고 오므로 런타임엔 가져오지 않는다
    from argos.brain.event_deep_analysis import DeepAnalysis

__all__ = [
    "DeepAnalysisSection",
    "DeepAnalysisView",
    "claim_deep_analysis",
    "clear_status",
    "load_deep_analysis_view",
    "mark_running",
    "save_result",
]


@dataclass(frozen=True)
class DeepAnalysisSection:
    heading: str
    body: str


@dataclass(frozen=True)
class DeepAnalysisView:
    event_id: uuid.UUID
    sections: tuple[DeepAnalysisSection, ...]
    analyzed_at: datetime | None
    doc_count: int | None
    status: DeepAnalysisStatus | None
    can_start: bool
    failure_message: str | None = None
    stale_notice: str | None = None

    @property
    def in_progress(self) -> bool:
        return self.status in (DeepAnalysisStatus.WAITING, DeepAnalysisStatus.RUNNING)


def _view_from_row(
    row: EventDeepAnalysis | None,
    event_id: uuid.UUID,
    current_doc_ids: Sequence[uuid.UUID],
) -> DeepAnalysisView:
    """행(없을 수 있음) → 화면용 뷰. DB를 건드리지 않는다."""
    sections: tuple[DeepAnalysisSection, ...] = ()
    analyzed_at = doc_count = status = None
    if row is not None:
        sections = tuple(
            DeepAnalysisSection(str(s.get("heading", "")), str(s.get("body", "")))
            for s in (row.sections or ())
        )
        analyzed_at = row.analyzed_at
        doc_count = row.doc_count
        status = row.status
    in_progress = status in (DeepAnalysisStatus.WAITING, DeepAnalysisStatus.RUNNING)
    return DeepAnalysisView(
        event_id=event_id,
        sections=sections,
        analyzed_at=analyzed_at,
        doc_count=doc_count,
        status=status,
        can_start=len(current_doc_ids) > 0 and not in_progress,
    )


async def load_deep_analysis_view(
    session: AsyncSession,
    event_id: uuid.UUID,
    current_doc_ids: Sequence[uuid.UUID],
) -> DeepAnalysisView:
    row = (
        await session.execute(
            select(EventDeepAnalysis).where(EventDeepAnalysis.event_id == event_id)
        )
    ).scalar_one_or_none()
    return _view_from_row(row, event_id, current_doc_ids)


_CLAIM_SQL = text(
    """
    INSERT INTO event_deep_analyses
        (id, event_id, status, requested_at, started_at, failure_reason,
         created_at, updated_at)
    VALUES
        (gen_random_uuid(), :event_id, CAST(:status AS deep_analysis_status),
         now(), NULL, NULL, now(), now())
    ON CONFLICT (event_id) DO UPDATE
        SET status = CAST(:status AS deep_analysis_status),
            requested_at = now(),
            started_at = NULL,
            failure_reason = NULL,
            updated_at = now()
        WHERE event_deep_analyses.status IS NULL
           OR event_deep_analyses.status = 'Failed'
    RETURNING id
    """
)


async def claim_deep_analysis(
    session: AsyncSession,
    event_id: uuid.UUID,
    *,
    status: DeepAnalysisStatus = DeepAnalysisStatus.RUNNING,
) -> bool:
    """진행 칸을 원자적으로 차지한다. 이미 대기·진행 중이면 False.

    결과 칸(sections 등)은 건드리지 않는다 — 다시 분석이 실패해도 직전 결과가 남는다.
    """
    row = (
        await session.execute(
            _CLAIM_SQL, {"event_id": event_id, "status": status.value}
        )
    ).first()
    await session.commit()
    return row is not None


async def mark_running(session: AsyncSession, event_id: uuid.UUID) -> None:
    await session.execute(
        update(EventDeepAnalysis)
        .where(EventDeepAnalysis.event_id == event_id)
        .values(status=DeepAnalysisStatus.RUNNING, started_at=func.now())
    )
    await session.commit()


async def save_result(
    session: AsyncSession, event_id: uuid.UUID, analysis: DeepAnalysis
) -> None:
    await session.execute(
        update(EventDeepAnalysis)
        .where(EventDeepAnalysis.event_id == event_id)
        .values(
            sections=analysis.sections_json(),
            analyzed_at=func.now(),
            doc_count=analysis.doc_count,
            doc_ids=list(analysis.doc_ids),
            model=analysis.model,
            status=None,
            failure_reason=None,
        )
    )
    await session.commit()


async def clear_status(session: AsyncSession, event_id: uuid.UUID) -> None:
    await session.execute(
        update(EventDeepAnalysis)
        .where(EventDeepAnalysis.event_id == event_id)
        .values(status=None, failure_reason=None)
    )
    await session.commit()
