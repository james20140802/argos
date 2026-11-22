"""ARG-333 — event_deep_analyses ORM 모델 (DB 불필요)."""
from __future__ import annotations

from argos.models import DeepAnalysisStatus, EventDeepAnalysis

_RESULT_COLUMNS = ("sections", "analyzed_at", "doc_count", "doc_ids", "model")
_PROGRESS_COLUMNS = ("status", "requested_at", "started_at", "failure_reason")


def _columns():
    return EventDeepAnalysis.__table__.c


def test_status_values_are_pascal_case():
    assert [s.value for s in DeepAnalysisStatus] == ["Waiting", "Running", "Failed"]
    assert list(_columns().status.type.enums) == ["Waiting", "Running", "Failed"]
    assert _columns().status.type.name == "deep_analysis_status"


def test_one_row_per_event_and_deleted_with_the_event():
    column = _columns().event_id
    assert column.nullable is False
    assert column.unique is True
    (fk,) = column.foreign_keys
    assert fk.column.table.name == "tech_events"
    assert fk.ondelete == "CASCADE"


def test_result_and_progress_can_each_be_empty_on_their_own():
    # 한 번도 성공한 적 없는 행(진행 상태만), 진행 중인 게 없는 행(결과만) 둘 다 가능해야 한다.
    for name in (*_RESULT_COLUMNS, *_PROGRESS_COLUMNS):
        assert _columns()[name].nullable is True, name


def test_only_adds_its_own_table_columns():
    expected = {"id", "event_id", "created_at", "updated_at", *_RESULT_COLUMNS, *_PROGRESS_COLUMNS}
    assert set(_columns().keys()) == expected


def test_failed_reanalysis_keeps_previous_result_in_memory():
    row = EventDeepAnalysis(
        sections=[{"heading": "성숙도", "body": "연구 프리뷰"}],
        doc_count=2,
        model="gemma4:26b",
    )
    row.status = DeepAnalysisStatus.FAILED
    row.failure_reason = "시간이 초과됐습니다"
    assert row.sections == [{"heading": "성숙도", "body": "연구 프리뷰"}]
    assert row.doc_count == 2
