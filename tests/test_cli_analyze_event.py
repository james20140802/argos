"""``argos analyze-event`` — 사건 심층 분석을 만들어 출력만 한다, 저장하지 않는다 (ARG-335). DB 없이."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos import cli
from argos.brain import event_deep_analysis as eda
from argos.brain import llm_client
from argos.brain.event_deep_analysis import (
    DEEP_ANALYSIS_HEADINGS,
    FAILURE_MESSAGES,
    DeepAnalysis,
    DeepAnalysisError,
    DeepAnalysisFailure,
    DeepAnalysisSection,
)

EVENT_ID = "00000000-0000-0000-0000-0000000000aa"


def _analysis() -> DeepAnalysis:
    return DeepAnalysis(
        sections=tuple(
            DeepAnalysisSection(h, f"{h}에 대한 본문") for h in DEEP_ANALYSIS_HEADINGS
        ),
        doc_ids=(uuid.UUID(int=1), uuid.UUID(int=2)),
        doc_count=2,
        model="gemma4:26b",
    )


@pytest.fixture
def wired(monkeypatch):
    session = MagicMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.execute = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    analyze = AsyncMock(return_value=_analysis())
    monkeypatch.setattr(eda, "analyze_event", analyze)
    client = MagicMock()
    client.unload = AsyncMock()
    monkeypatch.setattr(llm_client, "get_deep_analysis_llm_client", lambda: client)
    analyze.client = client
    return session, analyze


def test_prints_five_headings_and_footer(wired, capsys):
    _, analyze = wired
    assert cli.main(["analyze-event", EVENT_ID]) == 0
    out = capsys.readouterr().out
    positions = [out.index(h) for h in DEEP_ANALYSIS_HEADINGS]
    assert positions == sorted(positions)
    assert "본문" in out
    assert out.strip().splitlines()[-1] == "기사 2건 · 모델 gemma4:26b"
    assert analyze.await_args.args[1] == uuid.UUID(EVENT_ID)


@pytest.mark.parametrize("reason", list(DeepAnalysisFailure))
def test_failure_prints_reason_and_exits_1(wired, capsys, reason):
    _, analyze = wired
    analyze.side_effect = DeepAnalysisError(reason, "detail")
    assert cli.main(["analyze-event", EVENT_ID]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip().splitlines()[-1] == f"분석 실패: {FAILURE_MESSAGES[reason]}"
    assert captured.out == ""


def test_failure_messages_are_the_agreed_wording():
    assert FAILURE_MESSAGES == {
        DeepAnalysisFailure.NoEvidence: "이 사건에 묶인 기사가 없습니다",
        DeepAnalysisFailure.ModelUnavailable: "모델이 응답하지 않습니다",
        DeepAnalysisFailure.Timeout: "시간이 초과됐습니다",
        DeepAnalysisFailure.BadFormat: "모델이 정해진 형식으로 답하지 않았습니다",
        DeepAnalysisFailure.TooLarge: "기사가 너무 많아 한 번에 읽지 못했습니다",
    }


def test_invalid_uuid_is_argparse_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze-event", "nope"])
    assert exc.value.code == 2


def test_writes_nothing(wired):
    session, _ = wired
    cli.main(["analyze-event", EVENT_ID])
    session.commit.assert_not_awaited()
    session.add.assert_not_called()
    session.execute.assert_not_awaited()


def test_unload_runs_after_success(wired):
    _, analyze = wired
    assert cli.main(["analyze-event", EVENT_ID]) == 0
    analyze.client.unload.assert_awaited_once_with("large")


def test_unload_skipped_when_no_evidence(wired):
    _, analyze = wired
    analyze.side_effect = DeepAnalysisError(DeepAnalysisFailure.NoEvidence)
    assert cli.main(["analyze-event", EVENT_ID]) == 1
    analyze.client.unload.assert_not_awaited()


def test_unload_still_runs_for_other_failures(wired):
    _, analyze = wired
    analyze.side_effect = DeepAnalysisError(DeepAnalysisFailure.Timeout)
    assert cli.main(["analyze-event", EVENT_ID]) == 1
    analyze.client.unload.assert_awaited_once_with("large")
