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
    "FAILURE_TEXT",
    "fail_interrupted_runs",
    "failure_code",
    "failure_message",
    "mark_failed",
    "load_deep_analysis_view",
    "mark_running",
    "mark_waiting",
    "save_result",
    "stale_notice",
    "WAITING_TEXT",
]

WAITING_TEXT = "대기 중(다른 작업이 모델 사용 중)"
RUNNING_TEXT = "분석 중…"

# failure_reason 칸에 저장하는 코드 → 화면 문구. 모델·예외의 원문은 화면에 내지 않는다.
FAILURE_TEXT: dict[str, str] = {
    "ModelUnavailable": "모델에 연결하지 못했어요",
    "ModelMissing": "분석 모델이 설치돼 있지 않아요",
    "Timeout": "시간 안에 분석을 마치지 못했어요",
    "BadFormat": "모델 답이 정해진 형식이 아니었어요",
    "TooLarge": "기사가 너무 많아 한 번에 읽지 못했어요",
    "NoEvidence": "이 사건에 묶인 기사가 없어요",
    "Interrupted": "웹이 다시 시작돼 분석이 중단됐어요",
    "Unknown": "알 수 없는 오류로 분석하지 못했어요",
}


def failure_code(exc: BaseException) -> str:
    """예외 → 저장용 실패 코드."""
    # argos.brain은 import 시 argos.database를 끌고 오므로 호출 시점에만 가져온다.
    from argos.brain.event_deep_analysis import DeepAnalysisError, DeepAnalysisFailure

    if not isinstance(exc, DeepAnalysisError):
        return "Unknown"
    if exc.reason is DeepAnalysisFailure.ModelUnavailable:
        detail = (exc.detail or "").lower()
        if "404" in detail or "not found" in detail:
            return "ModelMissing"
    return exc.reason.value


def failure_message(code: str | None) -> str | None:
    if code is None:
        return None
    return FAILURE_TEXT.get(code, FAILURE_TEXT["Unknown"])


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

    @property
    def progress_text(self) -> str | None:
        """진행 중일 때 화면에 낼 문구. 진행 중이 아니면 None."""
        if self.status == DeepAnalysisStatus.WAITING:
            return WAITING_TEXT
        if self.status == DeepAnalysisStatus.RUNNING:
            return RUNNING_TEXT
        return None

    @property
    def stale_lead(self) -> str:
        """안내 문구에서 버튼이 되는 끝의 '다시 분석'을 뺀 앞부분."""
        return (self.stale_notice or "").removesuffix("다시 분석")


_STALE_ACTION = "다시 분석"


def stale_notice(
    stored_doc_ids: Sequence[uuid.UUID] | None,
    current_doc_ids: Sequence[uuid.UUID],
) -> str | None:
    """분석 당시 기사 묶음과 지금 묶음을 비교해 안내 문구를 만든다. 같으면 None."""
    if stored_doc_ids is None:
        return None
    stored, current = set(stored_doc_ids), set(current_doc_ids)
    added = len(current - stored)
    if added:
        return f"새 기사 {added}건이 더 붙었어요 · {_STALE_ACTION}"
    if stored - current:
        return f"기사 묶음이 달라졌어요 · {_STALE_ACTION}"
    return None


def _view_from_row(
    row: EventDeepAnalysis | None,
    event_id: uuid.UUID,
    current_doc_ids: Sequence[uuid.UUID],
) -> DeepAnalysisView:
    """행(없을 수 있음) → 화면용 뷰. DB를 건드리지 않는다."""
    sections: tuple[DeepAnalysisSection, ...] = ()
    analyzed_at = doc_count = status = None
    message: str | None = None
    if row is not None:
        sections = tuple(
            DeepAnalysisSection(str(s.get("heading", "")), str(s.get("body", "")))
            for s in (row.sections or ())
        )
        analyzed_at = row.analyzed_at
        doc_count = row.doc_count
        status = row.status
        if status == DeepAnalysisStatus.FAILED:
            message = failure_message(row.failure_reason or "Unknown")
    in_progress = status in (DeepAnalysisStatus.WAITING, DeepAnalysisStatus.RUNNING)
    return DeepAnalysisView(
        event_id=event_id,
        sections=sections,
        analyzed_at=analyzed_at,
        doc_count=doc_count,
        status=status,
        can_start=len(current_doc_ids) > 0 and not in_progress,
        failure_message=message,
        stale_notice=(
            stale_notice(row.doc_ids or (), current_doc_ids)
            if row is not None and row.sections and not in_progress
            else None
        ),
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


async def mark_waiting(session: AsyncSession, event_id: uuid.UUID) -> None:
    """다른 작업이 순번을 쥐고 있어 기다리는 중임을 기록한다.

    started_at은 mark_running만 채우므로, 비어 있는 동안(claim이 Running으로 잡았어도)에는 대기로
    바꾸고, 실제로 시작한 뒤에는 건드리지 않는다. Failed·없는 행도 건드리지 않는다.
    """
    await session.execute(
        update(EventDeepAnalysis)
        .where(
            EventDeepAnalysis.event_id == event_id,
            EventDeepAnalysis.started_at.is_(None),
            EventDeepAnalysis.status.in_(
                (DeepAnalysisStatus.WAITING, DeepAnalysisStatus.RUNNING)
            ),
        )
        .values(status=DeepAnalysisStatus.WAITING)
    )
    await session.commit()


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


async def mark_failed(session: AsyncSession, event_id: uuid.UUID, code: str) -> None:
    """실패를 기록한다. 결과 칸은 건드리지 않아 직전 성공 결과가 남는다."""
    await session.execute(
        update(EventDeepAnalysis)
        .where(EventDeepAnalysis.event_id == event_id)
        .values(status=DeepAnalysisStatus.FAILED, failure_reason=code)
    )
    await session.commit()


async def fail_interrupted_runs(session: AsyncSession) -> int:
    """웹 시작 시: 대기·진행 중으로 남은 행(이전 웹 프로세스의 잔재)을 '중단됨'으로 정리."""
    result = await session.execute(
        update(EventDeepAnalysis)
        .where(
            EventDeepAnalysis.status.in_(
                (DeepAnalysisStatus.WAITING, DeepAnalysisStatus.RUNNING)
            )
        )
        .values(status=DeepAnalysisStatus.FAILED, failure_reason="Interrupted")
    )
    await session.commit()
    return result.rowcount or 0
