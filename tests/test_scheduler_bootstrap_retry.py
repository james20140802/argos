"""`argos schedule install` 재등록 경쟁·부분 실패 (2026-10-25 운영 재현) — launchctl은 가짜.

운영에서 KeepAlive 데몬 com.argos.web 을 bootout 직후 bootstrap 하면
``exit 5: Input/output error`` 로 실패했고, 그 뒤 job(correct, classify)은
다시 올라가지 않았다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from argos import scheduler
from argos.scheduler import SchedulerError, bootstrap_plist, install_plist, reload_schedule


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    binary = tmp_path / "argos"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", tmp_path / "LaunchAgents")
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", tmp_path / "Logs")
    xdg = tmp_path / "xdg"
    (xdg / "argos").mkdir(parents=True)
    (xdg / "argos" / ".env").write_text("POSTGRES_USER=argos\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("ARGOS_ENV_FILE", raising=False)


class FakeLaunchd:
    """bootout 뒤에도 ``linger`` 번의 print 동안 서비스가 남아 있는 launchd.

    남아 있는 동안의 bootstrap 은 운영에서 본 exit 5 로 실패한다.
    ``always_fail`` 에 든 label 은 매번 실패한다.
    """

    def __init__(self, *, loaded=(), linger: int = 0, always_fail=()):
        self.loaded = set(loaded)
        self.tearing_down: dict[str, int] = {}
        self.linger = linger
        self.always_fail = set(always_fail)
        self.calls: list[tuple[str, str]] = []
        self.sleeps: list[float] = []

    def _io_error(self, args):
        return subprocess.CompletedProcess(
            ["launchctl", *args], 5, stdout="",
            stderr="Bootstrap failed: 5: Input/output error",
        )

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        action = args[0]
        if action == "bootstrap":
            label = Path(args[2]).stem
            self.calls.append(("bootstrap", label))
            if label in self.always_fail or label in self.tearing_down:
                return self._io_error(args)
            self.loaded.add(label)
            return subprocess.CompletedProcess(["launchctl", *args], 0, stdout="", stderr="")
        label = args[1].rsplit("/", 1)[-1]
        self.calls.append((action, label))
        if action == "bootout":
            if label not in self.loaded:
                return subprocess.CompletedProcess(
                    ["launchctl", *args], 36, stdout="",
                    stderr="Could not find specified service",
                )
            self.loaded.discard(label)
            if self.linger:
                self.tearing_down[label] = self.linger
            return subprocess.CompletedProcess(["launchctl", *args], 0, stdout="", stderr="")
        if action == "print":
            if label in self.tearing_down:
                self.tearing_down[label] -= 1
                if self.tearing_down[label] <= 0:
                    del self.tearing_down[label]
                return subprocess.CompletedProcess(
                    ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
                )
            if label in self.loaded:
                return subprocess.CompletedProcess(
                    ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
                )
            return subprocess.CompletedProcess(
                ["launchctl", *args], 113, stdout="", stderr="Could not find service"
            )
        raise AssertionError(f"unexpected launchctl call: {args}")


def _install(monkeypatch, fake: FakeLaunchd) -> None:
    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    monkeypatch.setattr(scheduler, "_sleep", fake.sleeps.append)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(time="06:00"),
        briefing=SimpleNamespace(time="07:00", weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"]),
        web=SimpleNamespace(launchd_enabled=True),
        correction=SimpleNamespace(enabled=True, time="03:00"),
        taxonomy=SimpleNamespace(assign_enabled=True, assign_time="04:30"),
    )


def test_bootstrap_waits_until_bootout_completes(monkeypatch, tmp_path):
    plist = tmp_path / "com.argos.web.plist"
    install_plist(plist, scheduler.render_web_plist(log_dir=tmp_path))
    fake = FakeLaunchd(loaded={"com.argos.web"}, linger=3)
    _install(monkeypatch, fake)

    bootstrap_plist(plist)

    # bootout → print 가 실패할 때까지 폴링 → 그 다음에야 bootstrap (첫 시도 성공).
    first_bootstrap = fake.calls.index(("bootstrap", "com.argos.web"))
    polls = fake.calls[1:first_bootstrap]
    assert fake.calls[0] == ("bootout", "com.argos.web")
    assert polls == [("print", "com.argos.web")] * 4
    assert fake.calls.count(("bootstrap", "com.argos.web")) == 1
    assert "com.argos.web" in fake.loaded
    assert len(fake.sleeps) == 3


def test_web_bootstrap_first_attempt_fails_then_succeeds_after_wait(monkeypatch):
    fake = FakeLaunchd(loaded={"com.argos.web"})
    _install(monkeypatch, fake)
    # 폴링이 "사라졌다"고 본 뒤에도 launchd 가 아직 정리 중인 경우 — 첫 bootstrap 은
    # exit 5, 잠시 기다린 다음 시도는 성공해야 한다.
    base = fake.__call__
    web_attempts = {"n": 0}

    def flaky(args):
        if args[0] == "bootstrap" and Path(args[2]).stem == "com.argos.web":
            web_attempts["n"] += 1
            if web_attempts["n"] == 1:
                fake.calls.append(("bootstrap", "com.argos.web"))
                return fake._io_error(args)
        return base(args)

    monkeypatch.setattr(scheduler, "_run_launchctl", flaky)

    reload_schedule(_config())

    assert web_attempts["n"] == 2
    assert "com.argos.web" in fake.loaded
    assert fake.sleeps, "retry must wait before the second bootstrap"
    assert {"com.argos.correct", "com.argos.classify"} <= fake.loaded


def test_one_job_failure_does_not_stop_the_rest(monkeypatch):
    fake = FakeLaunchd(
        loaded={"com.argos.web", "com.argos.correct", "com.argos.classify"},
        always_fail={"com.argos.web"},
    )
    _install(monkeypatch, fake)

    with pytest.raises(SchedulerError) as exc_info:
        reload_schedule(_config())

    # web 뒤에 오는 job 도 끝까지 다시 올린다.
    assert {
        "com.argos.run",
        "com.argos.brief",
        "com.argos.brief-weekly",
        "com.argos.correct",
        "com.argos.classify",
    } <= fake.loaded
    assert ("bootstrap", "com.argos.classify") in fake.calls
    msg = str(exc_info.value)
    assert "1 of 6" in msg
    assert "com.argos.web" in msg and "Input/output error" in msg
    loaded_line = msg.split("Loaded:", 1)[1]
    assert "com.argos.classify" in loaded_line
    assert "com.argos.web" not in loaded_line


def test_every_failure_is_listed(monkeypatch):
    fake = FakeLaunchd(always_fail={"com.argos.brief", "com.argos.web"})
    _install(monkeypatch, fake)

    with pytest.raises(SchedulerError) as exc_info:
        reload_schedule(_config())

    msg = str(exc_info.value)
    assert "2 of 6" in msg
    assert "  - com.argos.brief:" in msg
    assert "  - com.argos.web:" in msg
    assert "com.argos.classify" in fake.loaded
