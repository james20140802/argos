"""argos correct-events — 수동 교정 실행·무쓰기 미리보기 (ARG-297). DB를 쓰지 않는다.

DB가 필요한 경로(후보 계산·반영)는 모듈 속성 훅을 가짜로 바꿔 끊는다.
"""
from __future__ import annotations

import uuid

import pytest

import argos.cli as cli_module
from argos.brain.event_correction import (
    AppliedMerge,
    AppliedSplit,
    CorrectionResult,
    SkippedCandidate,
)
from argos.brain.graph_backend import GraphLibsUnavailable
from argos.brain.recluster_candidates import (
    MergeCandidate,
    ReclusterCandidates,
    SplitCandidate,
)
from argos.cli import _graph_install_commands, main

D1, D2 = uuid.UUID(int=1), uuid.UUID(int=2)
E1, E2, E3, E4, E5 = (uuid.UUID(int=100 + i) for i in range(1, 6))

PERIOD = ["--from", "2026-10-01", "--to", "2026-10-10"]


def _candidates() -> ReclusterCandidates:
    return ReclusterCandidates(
        merges=(MergeCandidate(event_ids=(E1, E2), evidence_document_ids=(D1, D2)),),
        splits=(SplitCandidate(event_id=E1, groups=((D1,), (D2,))),),
    )


@pytest.fixture
def fake_recluster(monkeypatch):
    seen: dict[str, object] = {"calls": 0}

    def _set(result=None, error: Exception | None = None):
        async def _fake(session, *, start, end):
            seen["calls"] += 1
            seen["start"], seen["end"] = start, end
            if error is not None:
                raise error
            return result

        monkeypatch.setattr(cli_module, "_recluster_period_for_cli", _fake)
        return seen

    return _set


@pytest.fixture
def fake_apply(monkeypatch):
    seen: dict[str, object] = {"calls": 0}

    def _set(result=None, error: Exception | None = None):
        async def _fake(candidates):
            seen["calls"] += 1
            seen["candidates"] = candidates
            if error is not None:
                raise error
            return result

        monkeypatch.setattr(cli_module, "_apply_corrections_for_cli", _fake)
        return seen

    return _set


def test_dry_run_prints_preview_and_never_applies(fake_recluster, fake_apply, capsys):
    fake_recluster(_candidates())
    applied = fake_apply(error=AssertionError("dry-run must not apply"))
    assert main(["correct-events", "--dry-run", *PERIOD]) == 0
    out = capsys.readouterr().out
    assert "합칠 후보: 1건" in out
    assert "가를 후보: 1건" in out
    assert "read-only" in out
    assert applied["calls"] == 0


def test_recluster_events_still_read_only(fake_recluster, fake_apply, capsys):
    fake_recluster(_candidates())
    applied = fake_apply(error=AssertionError("recluster-events must not apply"))
    assert main(["recluster-events", *PERIOD]) == 0
    assert "recluster-events (read-only)" in capsys.readouterr().out
    assert applied["calls"] == 0


def test_apply_passes_period_and_prints_applied_and_skipped(
    fake_recluster, fake_apply, capsys
):
    recluster = fake_recluster(_candidates())
    applied = fake_apply(
        CorrectionResult(
            merges=[AppliedMerge(E1, (E2,), 3)],
            splits=[AppliedSplit(E3, (E4,))],
            skipped=[SkippedCandidate("split", (E5,), "capped")],
        )
    )
    assert main(["correct-events", *PERIOD]) == 0
    out = capsys.readouterr().out
    assert recluster["start"].date().isoformat() == "2026-10-01"
    assert recluster["end"].date().isoformat() == "2026-10-10"
    assert applied["calls"] == 1
    assert "합침 반영: 1건" in out
    assert "가름 반영: 1건" in out
    assert "건너뜀: 1건" in out
    assert "capped" in out
    assert str(E1) in out


def test_apply_failure_exits_nonzero_without_traceback(
    fake_recluster, fake_apply, capsys
):
    fake_recluster(_candidates())
    fake_apply(error=RuntimeError("boom"))
    assert main(["correct-events", *PERIOD]) == 1
    captured = capsys.readouterr()
    assert "boom" in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_compute_failure_exits_nonzero(fake_recluster, fake_apply, capsys):
    fake_recluster(error=OSError("db down"))
    applied = fake_apply(CorrectionResult())
    assert main(["correct-events", *PERIOD]) == 1
    captured = capsys.readouterr()
    assert "db down" in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err
    assert applied["calls"] == 0


def test_missing_graph_libs_warns_and_exits_zero(fake_recluster, fake_apply, capsys):
    fake_recluster(error=GraphLibsUnavailable())
    applied = fake_apply(CorrectionResult())
    assert main(["correct-events", *PERIOD]) == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.out + captured.err
    assert _graph_install_commands() in captured.out + captured.err
    assert applied["calls"] == 0


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_end_before_start_is_rejected(extra, fake_recluster, fake_apply, capsys):
    recluster = fake_recluster(_candidates())
    applied = fake_apply(CorrectionResult())
    code = main(
        ["correct-events", *extra, "--from", "2026-10-10", "--to", "2026-10-01"]
    )
    assert code == 2
    assert "--to" in capsys.readouterr().err
    assert recluster["calls"] == 0
    assert applied["calls"] == 0
