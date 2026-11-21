"""CLI가 모델을 쓰는 명령 전체를 모델 순번으로 감싼다 (ARG-334)."""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from argos import cli
from argos.brain import model_turn as mt
from argos.brain.model_turn import LOCK_ENV, WAIT_MESSAGE, turn_is_held
from argos.config import settings

_HOLDER = """
import fcntl, sys, time
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX)
print("ready", flush=True)
time.sleep(60)
"""


@pytest.fixture
def spawn_holder(tmp_path):
    procs: list[subprocess.Popen] = []

    def spawn() -> subprocess.Popen:
        import os

        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, os.environ[LOCK_ENV]],
            stdout=subprocess.PIPE,
            text=True,
        )
        procs.append(proc)
        assert proc.stdout.readline().strip() == "ready"
        return proc

    yield spawn
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(mt, "DEFAULT_POLL_SECONDS", 0.01)


def _job(argv: list[str]) -> str | None:
    return cli._model_turn_job(cli._build_parser().parse_args(argv))


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["run"], "run"),
        (["add", "https://x"], "add"),
        # taxonomy
        (["taxonomy", "assign"], "taxonomy assign"),
        (["taxonomy", "assign", "--dry-run"], "taxonomy assign"),
        (["taxonomy", "draft"], "taxonomy draft"),
        (["taxonomy", "draft", "--seed"], None),
        (["taxonomy", "eval", "--tree", "spike"], "taxonomy eval"),
        (["taxonomy", "eval"], None),
        (["taxonomy", "show"], None),
        (["taxonomy", "export"], None),
        (["taxonomy", "rename", "a", "b"], None),
        (["taxonomy", "merge", "a", "b"], None),
        (["taxonomy", "split", "a"], None),
        (["taxonomy", "delete", "a"], None),
        (["taxonomy", "apply"], None),
        (["taxonomy", "apply", "--dry-run"], None),
        (["taxonomy", "why", "x"], None),
        # 코드 확인 행
        (["resummarize-events"], "resummarize-events"),
        (["resummarize-events", "--dry-run"], None),
        (["backfill-trust"], "backfill-trust"),
        (["backfill-trust", "--dry-run"], None),
        (["backfill-events"], None),
        (["backfill-events", "--dry-run"], None),
        (["backfill-events", "--rename-stale"], "backfill-events"),
        (["backfill-events", "--rename-stale", "--dry-run"], None),
        (["correct-events"], "correct-events"),
        (["alias-names"], None),
        (["alias-names", "--dry-run"], None),
        (["backfill-entities"], None),
        (["backfill-entities", "--dry-run"], None),
        (["brief", "--weekly"], None),
        # 비대상
        (["search", "q"], None),
        (["slack"], None),
        (["web"], None),
        (["brief"], None),
        (["portfolio"], None),
        (["stats"], None),
        (["status"], None),
        (["doctor"], None),
        (["backfill-images"], None),
        (["body-stats"], None),
        (["refetch-bodies"], None),
        (["backfill-digests"], None),
        (["schedule", "status"], None),
        (["config", "list"], None),
    ],
)
def test_model_turn_job_table(argv, expected):
    assert _job(argv) == expected


@pytest.mark.parametrize("command", ["recluster-events", "correct-events"])
def test_pair_check_decides_recluster_family_dry_runs(monkeypatch, command):
    extra = ["--dry-run"] if command == "correct-events" else []
    monkeypatch.setattr(cli, "_pair_check_may_call_llm", lambda args: True)
    assert _job([command, *extra]) == command
    monkeypatch.setattr(cli, "_pair_check_may_call_llm", lambda args: False)
    assert _job([command, *extra]) is None


def test_pair_check_predicate_follows_config_and_flag(monkeypatch):
    cfg = settings.user.event_detection
    monkeypatch.setattr(cfg, "pair_check_enabled", True)
    monkeypatch.setattr(cfg, "pair_check_max_calls", 300)
    parse = cli._build_parser().parse_args
    assert cli._pair_check_may_call_llm(parse(["recluster-events"])) is True
    assert (
        cli._pair_check_may_call_llm(parse(["recluster-events", "--pair-check-max-calls", "0"]))
        is False
    )
    monkeypatch.setattr(cfg, "pair_check_max_calls", 0)
    assert cli._pair_check_may_call_llm(parse(["recluster-events"])) is False
    assert (
        cli._pair_check_may_call_llm(parse(["recluster-events", "--pair-check-max-calls", "5"]))
        is True
    )
    monkeypatch.setattr(cfg, "pair_check_enabled", False)
    assert (
        cli._pair_check_may_call_llm(parse(["recluster-events", "--pair-check-max-calls", "5"]))
        is False
    )


def test_target_command_holds_turn_for_its_whole_run(monkeypatch):
    seen: list[bool] = []

    def body(args):
        # 별도 스레드(순번 밖 컨텍스트)에서 본다.
        result: list[bool] = []
        t = threading.Thread(target=lambda: result.append(turn_is_held()))
        t.start()
        t.join()
        seen.extend(result)
        return 0

    monkeypatch.setattr(cli, "_dispatch", body)
    assert cli.main(["run"]) == 0
    assert seen == [True]
    assert turn_is_held() is False


def test_non_target_command_runs_while_turn_is_held(monkeypatch, spawn_holder, capsys):
    spawn_holder()
    ran: list[bool] = []
    monkeypatch.setattr(cli, "_dispatch", lambda args: ran.append(True) or 0)
    assert cli.main(["search", "q"]) == 0
    assert ran == [True]
    assert WAIT_MESSAGE not in capsys.readouterr().err


def test_target_command_prints_wait_line_then_resumes(monkeypatch, spawn_holder, capsys):
    holder = spawn_holder()
    killer = threading.Timer(0.2, holder.kill)
    killer.start()
    ran: list[bool] = []
    monkeypatch.setattr(cli, "_dispatch", lambda args: ran.append(True) or 0)
    try:
        assert cli.main(["taxonomy", "assign", "--dry-run"]) == 0
    finally:
        killer.cancel()
        killer.join()

    lines = capsys.readouterr().err.splitlines()
    assert lines.count(WAIT_MESSAGE) == 1
    assert sum(line.startswith("대기 끝") for line in lines) == 1
    assert lines.index(WAIT_MESSAGE) < next(
        i for i, line in enumerate(lines) if line.startswith("대기 끝")
    )
    assert ran == [True]


def test_no_wait_lines_when_turn_is_free(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_dispatch", lambda args: 0)
    assert cli.main(["run"]) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure", ["raise", "nonzero"])
def test_turn_released_when_command_fails(monkeypatch, failure):
    def body(args):
        if failure == "raise":
            raise RuntimeError("boom")
        return 3

    monkeypatch.setattr(cli, "_dispatch", body)
    if failure == "raise":
        with pytest.raises(RuntimeError):
            cli.main(["run"])
    else:
        assert cli.main(["run"]) == 3
    assert turn_is_held() is False
