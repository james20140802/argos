"""``argos resummarize-events`` — 몇 건 바꿀지 먼저 보여 주고 확인 후에만 쓴다 (ARG-308). DB 없이."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos import cli
from argos.brain import event_backfill, llm_client
from argos.brain.event_backfill import RenameResult, StaleEvent
from argos.brain.event_naming import EvidenceDoc


def _events() -> list[StaleEvent]:
    return [
        StaleEvent(
            event_id=uuid.uuid4(),
            docs=[EvidenceDoc(title="a", body="x"), EvidenceDoc(title="b", body="y")],
            title="옛 사건 제목",
        ),
        StaleEvent(event_id=uuid.uuid4(), docs=[EvidenceDoc(title="c")], title=None),
    ]


@pytest.fixture
def wired(monkeypatch):
    session = MagicMock()
    session.commit = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    fetch = AsyncMock(return_value=_events())
    monkeypatch.setattr(event_backfill, "fetch_stale_events", fetch)
    monkeypatch.setattr(event_backfill, "count_live_events", AsyncMock(return_value=40))
    rename = AsyncMock(return_value=RenameResult(renamed=2, skipped=0))
    monkeypatch.setattr(event_backfill, "rename_stale_events", rename)
    client = MagicMock()
    client.unload = AsyncMock()
    monkeypatch.setattr(llm_client, "get_event_summary_llm_client", lambda: client)
    return session, fetch, rename, client


def test_preview_shows_how_many_will_change_before_anything(wired, capsys, monkeypatch):
    session, fetch, rename, _ = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("dry-run must not ask"))
    assert cli.main(["resummarize-events", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "전체 40건 중 2건 다시 요약 (근거 기사 3건, 예상 ~2분)" in out
    assert "옛 사건 제목" in out and "(무명)" in out
    assert fetch.await_args.kwargs["include_unsectioned"] is True
    rename.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_declined_writes_nothing(wired, monkeypatch, capsys):
    session, _, rename, _ = wired
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert cli.main(["resummarize-events"]) != 0
    assert "취소되었습니다." in capsys.readouterr().out
    rename.assert_not_awaited()


def test_confirmed_runs_and_unloads_the_model(wired, monkeypatch):
    _, _, rename, client = wired
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert cli.main(["resummarize-events"]) == 0
    rename.assert_awaited_once()
    assert rename.await_args.kwargs["client"] is client
    client.unload.assert_awaited_once_with("large")


def test_yes_skips_the_prompt_and_limit_is_forwarded(wired, monkeypatch):
    _, fetch, rename, _ = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("--yes must not ask"))
    assert cli.main(["resummarize-events", "--yes", "--limit", "7"]) == 0
    assert fetch.await_args.kwargs["limit"] == 7
    rename.assert_awaited_once()


def test_skipped_events_exit_1(wired, monkeypatch, capsys):
    _, _, rename, _ = wired
    rename.return_value = RenameResult(renamed=1, skipped=1)
    assert cli.main(["resummarize-events", "--yes"]) == 1
    assert "이전 이름·요약을 유지" in capsys.readouterr().out


def test_nothing_to_do_does_not_ask(wired, monkeypatch, capsys):
    _, fetch, rename, _ = wired
    fetch.return_value = []
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not ask"))
    assert cli.main(["resummarize-events"]) == 0
    assert "다시 요약할 사건이 없습니다." in capsys.readouterr().out
    rename.assert_not_awaited()
