"""argos recluster-events — 인자 검증·안내·리포트 (ARG-281). DB를 쓰지 않는다.

DB가 필요한 경로는 오케스트레이션 함수를 가짜로 바꿔 끊는다. 실제 코퍼스로
도는 검증은 tests/brain/test_recluster_e2e_db.py가 맡는다.
"""
from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from argos.brain.graph_backend import GraphLibsUnavailable
from argos.brain.recluster_candidates import (
    MergeCandidate,
    ReclusterCandidates,
    SplitCandidate,
)
from argos.cli import main
from argos.config import settings

D1, D2, D3 = uuid.UUID(int=1), uuid.UUID(int=2), uuid.UUID(int=3)
E1, E2 = uuid.UUID(int=101), uuid.UUID(int=102)


@pytest.fixture
def fake_recluster(monkeypatch):
    """`recluster_period`를 원하는 결과로 바꾼다. 반환값은 setter."""

    import argos.cli as cli_module

    def _set(result):
        async def _fake(session, *, start, end):
            return result

        monkeypatch.setattr(cli_module, "_recluster_period_for_cli", _fake)

    return _set


def test_end_before_start_is_rejected_without_computing(capsys):
    code = main(["recluster-events", "--from", "2026-08-31", "--to", "2026-08-01"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--to" in captured.out + captured.err


def test_a_malformed_date_is_rejected(capsys):
    # argparse의 type= 오류는 SystemExit(2)로 나온다 — 레포의 기존 CLI 테스트
    # 관례와 같다 (tests/test_cli_search.py 참고).
    with pytest.raises(SystemExit) as exc_info:
        main(["recluster-events", "--from", "not-a-date", "--to", "2026-08-01"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "YYYY-MM-DD" in captured.out + captured.err


def test_missing_graph_libraries_print_the_install_hint_without_a_traceback(
    monkeypatch, capsys
):
    async def _raise(session, *, start, end):
        raise GraphLibsUnavailable()

    import argos.cli as cli_module

    monkeypatch.setattr(cli_module, "_recluster_period_for_cli", _raise)
    code = main(["recluster-events", "--from", "2026-08-01", "--to", "2026-08-31"])
    captured = capsys.readouterr()
    assert code == 1
    assert "uv sync --all-extras" in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_no_candidates_reports_none_and_exits_zero(fake_recluster, capsys):
    fake_recluster(ReclusterCandidates(merges=(), splits=()))
    code = main(["recluster-events", "--from", "2026-08-01", "--to", "2026-08-31"])
    captured = capsys.readouterr()
    assert code == 0
    assert "없음" in captured.out


def test_candidates_are_printed_in_a_readable_report(fake_recluster, capsys):
    fake_recluster(
        ReclusterCandidates(
            merges=(
                MergeCandidate(event_ids=(E1, E2), evidence_document_ids=(D1, D2)),
            ),
            splits=(SplitCandidate(event_id=E1, groups=((D1,), (D2, D3))),),
        )
    )
    code = main(["recluster-events", "--from", "2026-08-01", "--to", "2026-08-31"])
    captured = capsys.readouterr()
    assert code == 0
    assert str(E1) in captured.out
    assert str(E2) in captured.out
    assert str(D1) in captured.out  # 근거 문서를 볼 수 있어야 한다
    assert "1" in captured.out


def test_the_report_repeats_byte_for_byte(fake_recluster, capsys):
    fake_recluster(
        ReclusterCandidates(
            merges=(
                MergeCandidate(event_ids=(E1, E2), evidence_document_ids=(D1, D2)),
            ),
            splits=(),
        )
    )
    main(["recluster-events", "--from", "2026-08-01", "--to", "2026-08-31"])
    first = capsys.readouterr().out
    main(["recluster-events", "--from", "2026-08-01", "--to", "2026-08-31"])
    second = capsys.readouterr().out
    assert first == second


def test_the_period_defaults_to_the_configured_window(fake_recluster, capsys):
    fake_recluster(ReclusterCandidates(merges=(), splits=()))
    code = main(["recluster-events"])  # --from/--to 없이도 돌아야 한다
    assert code == 0
    assert "없음" in capsys.readouterr().out


@pytest.fixture
def recorded_period(monkeypatch):
    """`--from`/`--to`가 실제로 어떤 기간으로 풀렸는지 잡아 둔다."""

    import argos.cli as cli_module

    seen: dict[str, object] = {}

    async def _record(session, *, start, end):
        seen["start"], seen["end"] = start, end
        return ReclusterCandidates(merges=(), splits=())

    monkeypatch.setattr(cli_module, "_recluster_period_for_cli", _record)
    return seen


@pytest.mark.parametrize("window_days, expected_days", [(14.0, 14), (1.0, 1), (0.5, 1)])
def test_the_default_period_spans_exactly_the_configured_window(
    recorded_period, monkeypatch, window_days, expected_days
):
    # `--from`을 비우면 기간은 `window_days`만큼이어야 한다. 예전에는 start를
    # **자정 기준** end에서 빼 놓고 end만 그날 끝까지 늘려서, 기본 기간이
    # 늘 하루씩 길었다 — 14일 설정이 15일을 훑고, 0.5는 12시간이 아니라 36시간.
    # `--from`/`--to`가 날짜 단위라 기간도 온전한 날 수여야 한다.
    monkeypatch.setattr(
        settings.user.event_detection, "window_days", window_days, raising=False
    )
    assert main(["recluster-events", "--to", "2026-08-31"]) == 0

    span = recorded_period["end"] - recorded_period["start"]
    # 끝은 그날 23:59:59.999999라 딱 하루 모자란 1µs가 빠진다.
    assert span == timedelta(days=expected_days) - timedelta(microseconds=1)
