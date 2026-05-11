"""Tests for src/argos/scheduler.py (ARG-51).

All subprocess interactions are mocked via ``monkeypatch.setattr(
scheduler, "_run_launchctl", fake)``; no real launchctl invocations
ever fire from these tests.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from argos import scheduler
from argos.scheduler import (
    SchedulerError,
    _calendar_intervals,
    _parse_hhmm,
    _resolve_argos_binary,
    _weekday_to_launchd,
    bootout_plist,
    bootstrap_plist,
    install_plist,
    is_loaded,
    reload_schedule,
    render_brief_plist,
    render_run_plist,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_argos_binary(monkeypatch, tmp_path) -> Path:
    """Force `shutil.which("argos")` to return a path inside tmp_path."""
    binary = tmp_path / "argos"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: str(binary))
    return binary


@pytest.fixture
def fake_uid(monkeypatch) -> int:
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)
    return 501


@pytest.fixture
def isolated_paths(monkeypatch, tmp_path) -> dict[str, Path]:
    """Redirect default LaunchAgents + Logs paths into tmp_path."""
    la = tmp_path / "LaunchAgents"
    logs = tmp_path / "Logs"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", logs)
    return {"launch_agents": la, "logs": logs}


def _make_fake_launchctl(
    handlers: dict[str, Callable[[list[str]], subprocess.CompletedProcess[str]]],
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    """Build a `_run_launchctl` replacement that dispatches on the first arg."""

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        action = args[0] if args else ""
        if action in handlers:
            return handlers[action](args)
        return subprocess.CompletedProcess(["launchctl", *args], 0, stdout="", stderr="")

    return fake


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("6:00", (6, 0)),
        ("06:00", (6, 0)),
        ("00:00", (0, 0)),
        ("23:59", (23, 59)),
        ("12:30", (12, 30)),
    ],
)
def test_parse_hhmm_accepts_valid(value: str, expected: tuple[int, int]) -> None:
    assert _parse_hhmm(value) == expected


@pytest.mark.parametrize(
    "bad",
    ["24:00", "6", "06:60", "abc", "", "06:0", "-1:00", "06:00:00", "06.00"],
)
def test_parse_hhmm_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_hhmm(bad)


# ---------------------------------------------------------------------------
# Weekday mapping (the silent-shift risk flagged in the plan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Sun", 0),
        ("Mon", 1),
        ("Tue", 2),
        ("Wed", 3),
        ("Thu", 4),
        ("Fri", 5),
        ("Sat", 6),
    ],
)
def test_weekday_to_launchd_full_table(name: str, expected: int) -> None:
    assert _weekday_to_launchd(name) == expected


def test_weekday_to_launchd_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        _weekday_to_launchd("Funday")


def test_calendar_intervals_full_week_collapses_to_single_dict() -> None:
    out = _calendar_intervals(
        6, 0, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    assert isinstance(out, dict)
    assert out == {"Hour": 6, "Minute": 0}
    assert "Weekday" not in out  # plan-mandated absence


def test_calendar_intervals_none_collapses_to_single_dict() -> None:
    out = _calendar_intervals(6, 0, None)
    assert out == {"Hour": 6, "Minute": 0}


def test_calendar_intervals_partial_week_expands_to_list_of_dicts() -> None:
    out = _calendar_intervals(7, 30, ["Mon", "Tue", "Wed", "Thu", "Fri"])
    assert isinstance(out, list)
    assert len(out) == 5
    assert out[0] == {"Hour": 7, "Minute": 30, "Weekday": 1}
    assert out[-1] == {"Hour": 7, "Minute": 30, "Weekday": 5}
    for entry in out:
        assert "Weekday" in entry


# ---------------------------------------------------------------------------
# Argos binary resolution
# ---------------------------------------------------------------------------


def test_resolve_argos_binary_uses_which(monkeypatch, tmp_path) -> None:
    binary = tmp_path / "argos"
    binary.write_text("")
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: str(binary))
    assert _resolve_argos_binary() == binary


def test_resolve_argos_binary_falls_back_to_usr_local(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)
    fake = tmp_path / "argos"
    fake.write_text("")
    monkeypatch.setattr(scheduler, "_ARGOS_BINARY_FALLBACKS", (fake,))
    assert _resolve_argos_binary() == fake


def test_resolve_argos_binary_raises_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)
    missing = tmp_path / "nope" / "argos"
    monkeypatch.setattr(scheduler, "_ARGOS_BINARY_FALLBACKS", (missing,))
    with pytest.raises(SchedulerError, match="Could not locate"):
        _resolve_argos_binary()


# ---------------------------------------------------------------------------
# Regression: the fallback chain must include Apple Silicon Homebrew
# (/opt/homebrew/bin/argos) before /usr/local/bin/argos, since Argos's
# primary target is M1 Max and GUI/launchd contexts have a minimal PATH
# where `shutil.which` returns None even when argos is installed.
# (Codex PR #41 — PRRT_kwDOR4m8Js6BDg1w)
# ---------------------------------------------------------------------------


def test_resolve_argos_binary_finds_apple_silicon_homebrew(monkeypatch) -> None:
    """When `shutil.which` returns None and only /opt/homebrew/bin/argos
    is a real file, `_resolve_argos_binary` must return that path. This
    is the GUI/launchd-spawn scenario on M1: PATH is minimal so which()
    misses the binary even though it's installed via Homebrew.
    """
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)

    apple_silicon_path = Path("/opt/homebrew/bin/argos")
    real_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if self == apple_silicon_path:
            return True
        # Defer to the real implementation for everything else so we
        # don't accidentally spoof other Path.is_file callers.
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    assert _resolve_argos_binary() == apple_silicon_path


def test_resolve_argos_binary_prefers_apple_silicon_over_usr_local(
    monkeypatch,
) -> None:
    """If both /opt/homebrew/bin/argos and /usr/local/bin/argos exist,
    the Apple Silicon path must win — Argos targets M1.
    """
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)

    apple = Path("/opt/homebrew/bin/argos")
    intel = Path("/usr/local/bin/argos")
    real_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if self in (apple, intel):
            return True
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    assert _resolve_argos_binary() == apple


def test_resolve_argos_binary_error_lists_all_four_paths(monkeypatch) -> None:
    """When every fallback misses, the error message must mention all
    four probed locations so operators know exactly what was checked.
    """
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: None)
    # No is_file overrides — none of the real fallback paths should
    # exist under the test runner. Force them to be missing in case
    # the runner machine actually has argos installed.
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    expected_fragments = [
        "/opt/homebrew/bin/argos",
        "/usr/local/bin/argos",
        ".local/bin/argos",
        "shutil.which",
    ]
    with pytest.raises(SchedulerError) as exc_info:
        _resolve_argos_binary()
    msg = str(exc_info.value)
    for fragment in expected_fragments:
        assert fragment in msg, f"missing fragment {fragment!r} in error: {msg!r}"


# ---------------------------------------------------------------------------
# Plist rendering — round-trip via plistlib.loads
# ---------------------------------------------------------------------------


def test_render_run_plist_round_trip(fake_argos_binary: Path, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    xml = render_run_plist(time="06:00", log_dir=log_dir)
    data = plistlib.loads(xml.encode("utf-8"))
    assert data["Label"] == "com.argos.run"
    assert data["ProgramArguments"][0] == str(fake_argos_binary)
    assert data["ProgramArguments"][1] == "run"
    assert data["StartCalendarInterval"] == {"Hour": 6, "Minute": 0}
    assert data["StandardOutPath"].endswith("run.log")
    assert data["StandardErrorPath"].endswith("run.log")
    assert data["RunAtLoad"] is False
    assert data["KeepAlive"] is False
    assert "PATH" in data["EnvironmentVariables"]


def test_render_brief_plist_weekday_subset(
    fake_argos_binary: Path, tmp_path: Path
) -> None:
    log_dir = tmp_path / "logs"
    xml = render_brief_plist(
        time="07:00",
        weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        log_dir=log_dir,
    )
    data = plistlib.loads(xml.encode("utf-8"))
    assert data["Label"] == "com.argos.brief"
    assert data["ProgramArguments"][1] == "brief"
    intervals = data["StartCalendarInterval"]
    assert isinstance(intervals, list)
    weekdays = [entry["Weekday"] for entry in intervals]
    assert weekdays == [1, 2, 3, 4, 5]
    assert data["StandardOutPath"].endswith("brief.log")


def test_render_brief_plist_full_week_collapses(
    fake_argos_binary: Path, tmp_path: Path
) -> None:
    xml = render_brief_plist(
        time="07:00",
        weekdays=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        log_dir=tmp_path,
    )
    data = plistlib.loads(xml.encode("utf-8"))
    assert isinstance(data["StartCalendarInterval"], dict)
    assert "Weekday" not in data["StartCalendarInterval"]


def test_render_plist_with_config_path(
    fake_argos_binary: Path, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.toml"
    xml = render_run_plist(time="06:00", config_path=cfg, log_dir=tmp_path)
    data = plistlib.loads(xml.encode("utf-8"))
    assert data["ProgramArguments"] == [
        str(fake_argos_binary),
        "run",
        "--config",
        str(cfg),
    ]


# ---------------------------------------------------------------------------
# install_plist atomicity
# ---------------------------------------------------------------------------


def test_install_plist_writes_atomically(
    fake_argos_binary: Path, tmp_path: Path
) -> None:
    target = tmp_path / "agents" / "com.argos.run.plist"
    log_dir = tmp_path / "logs"
    xml = render_run_plist(time="06:00", log_dir=log_dir)
    install_plist(target, xml)
    assert target.exists()
    # 0o644
    assert (target.stat().st_mode & 0o777) == 0o644
    # Log dir was created
    assert log_dir.exists()
    # No leftover tmp
    assert not target.with_suffix(target.suffix + ".tmp").exists()


def test_install_plist_replace_failure_leaves_original_untouched(
    fake_argos_binary: Path, tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "com.argos.run.plist"
    log_dir = tmp_path / "logs"
    target.write_text("ORIGINAL")
    xml = render_run_plist(time="06:00", log_dir=log_dir)

    def boom(src, dst):  # noqa: ARG001
        raise OSError("simulated replace failure")

    monkeypatch.setattr(scheduler.os, "replace", boom)

    with pytest.raises(OSError, match="simulated replace failure"):
        install_plist(target, xml)

    assert target.read_text() == "ORIGINAL"
    assert not target.with_suffix(target.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# is_loaded / bootstrap_plist / bootout_plist
# ---------------------------------------------------------------------------


def test_is_loaded_true_when_print_succeeds_and_label_in_stdout(
    monkeypatch, fake_uid: int
) -> None:
    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args[0] == "print"
        assert args[1] == f"gui/{fake_uid}/com.argos.run"
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="com.argos.run = service\n", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    assert is_loaded("com.argos.run") is True


def test_is_loaded_false_when_print_fails(monkeypatch, fake_uid: int) -> None:
    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["launchctl", *args], 1, stdout="", stderr="Could not find specified service"
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    assert is_loaded("com.argos.run") is False


def test_is_loaded_false_when_label_missing_from_stdout(
    monkeypatch, fake_uid: int
) -> None:
    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="some other unrelated text", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    assert is_loaded("com.argos.run") is False


def test_bootstrap_plist_idempotent(
    monkeypatch, fake_argos_binary: Path, fake_uid: int, tmp_path: Path
) -> None:
    target = tmp_path / "com.argos.run.plist"
    install_plist(target, render_run_plist(time="06:00", log_dir=tmp_path))

    calls: list[list[str]] = []

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "bootout":
            # Already-loaded: success. Already-absent: tolerated.
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout="", stderr=""
            )
        if args[0] == "bootstrap":
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout="", stderr=""
            )
        if args[0] == "print":
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout="com.argos.run = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    bootstrap_plist(target)
    # bootout + bootstrap + print(is_loaded) at minimum
    actions = [c[0] for c in calls]
    assert "bootout" in actions
    assert "bootstrap" in actions
    assert "print" in actions


def test_bootstrap_plist_raises_on_failure(
    monkeypatch, fake_argos_binary: Path, fake_uid: int, tmp_path: Path
) -> None:
    target = tmp_path / "com.argos.run.plist"
    install_plist(target, render_run_plist(time="06:00", log_dir=tmp_path))

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "bootstrap":
            return subprocess.CompletedProcess(
                ["launchctl", *args],
                5,
                stdout="",
                stderr="Bootstrap failed: 5: Input/output error",
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    with pytest.raises(SchedulerError, match="bootstrap failed"):
        bootstrap_plist(target)


def test_bootstrap_plist_raises_when_post_check_fails(
    monkeypatch, fake_argos_binary: Path, fake_uid: int, tmp_path: Path
) -> None:
    target = tmp_path / "com.argos.run.plist"
    install_plist(target, render_run_plist(time="06:00", log_dir=tmp_path))

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            # is_loaded → False
            return subprocess.CompletedProcess(
                ["launchctl", *args], 1, stdout="", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    with pytest.raises(SchedulerError, match="not loaded"):
        bootstrap_plist(target)


def test_bootout_plist_tolerates_already_absent_exit_36(
    monkeypatch, fake_uid: int
) -> None:
    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["launchctl", *args],
            36,
            stdout="",
            stderr="Could not find specified service",
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    # Should not raise.
    bootout_plist("com.argos.run")


def test_bootout_plist_tolerates_text_hint(monkeypatch, fake_uid: int) -> None:
    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["launchctl", *args],
            3,
            stdout="",
            stderr="No such process",
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    bootout_plist("com.argos.run")  # no raise


def test_bootout_plist_raises_on_unexpected_failure(
    monkeypatch, fake_uid: int
) -> None:
    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["launchctl", *args],
            1,
            stdout="",
            stderr="Operation not permitted",
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    with pytest.raises(SchedulerError, match="bootout failed"):
        bootout_plist("com.argos.run")


# ---------------------------------------------------------------------------
# reload_schedule integration
# ---------------------------------------------------------------------------


def test_reload_schedule_writes_both_plists_and_bootstraps(
    monkeypatch,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
) -> None:
    user_config = SimpleNamespace(
        run=SimpleNamespace(time="06:00"),
        briefing=SimpleNamespace(
            time="07:00",
            weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        ),
    )

    bootstrap_calls: list[str] = []

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "bootstrap":
            bootstrap_calls.append(args[2])  # the plist path
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout="", stderr=""
            )
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    reload_schedule(user_config)

    run_plist = isolated_paths["launch_agents"] / "com.argos.run.plist"
    brief_plist = isolated_paths["launch_agents"] / "com.argos.brief.plist"
    assert run_plist.exists()
    assert brief_plist.exists()
    assert str(run_plist) in bootstrap_calls
    assert str(brief_plist) in bootstrap_calls


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_schedule_status(
    monkeypatch, capsys, fake_uid: int
) -> None:
    from argos.cli import main

    state: dict[str, bool] = {"com.argos.run": True, "com.argos.brief": False}

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            if state.get(label):
                return subprocess.CompletedProcess(
                    ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
                )
            return subprocess.CompletedProcess(
                ["launchctl", *args], 1, stdout="", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    rc = main(["schedule", "status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "com.argos.run: loaded" in out
    assert "com.argos.brief: not loaded" in out


def test_cli_schedule_install_invokes_reload(
    monkeypatch, capsys, fake_argos_binary: Path, fake_uid: int, isolated_paths
) -> None:
    from argos.cli import main

    called = {"reload": 0}

    def fake_reload(user_config, *, config_path=None) -> None:  # noqa: ARG001
        called["reload"] += 1

    monkeypatch.setattr(scheduler, "reload_schedule", fake_reload)
    # cli.py imports reload_schedule lazily, so also patch the module attr that
    # the CLI dispatcher binds to.
    import argos.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "reload_schedule", fake_reload)

    rc = main(["schedule", "install"])
    out = capsys.readouterr().out
    assert rc == 0
    assert called["reload"] == 1
    assert "com.argos.run" in out


def test_cli_schedule_uninstall_calls_bootout(
    monkeypatch, capsys, fake_uid: int
) -> None:
    from argos.cli import main

    bootout_labels: list[str] = []

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "bootout":
            bootout_labels.append(args[1].rsplit("/", 1)[-1])
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    rc = main(["schedule", "uninstall"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "com.argos.run" in bootout_labels
    assert "com.argos.brief" in bootout_labels
    assert "Unloaded: com.argos.run" in out


# ---------------------------------------------------------------------------
# Regression: reload_schedule forwards config_path to both plist renderers
# (Codex PR #41 — PRRT_kwDOR4m8Js6BC_SD)
# ---------------------------------------------------------------------------


def test_reload_schedule_forwards_config_path(
    monkeypatch,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Both rendered plists must embed `--config <path>` when reload_schedule
    is called with ``config_path=...`` — otherwise the scheduled jobs would
    silently run against the default config, not the operator-selected one.
    """
    user_config = SimpleNamespace(
        run=SimpleNamespace(time="06:00"),
        briefing=SimpleNamespace(
            time="07:00",
            weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        ),
    )
    cfg_path = tmp_path / "custom-config.toml"

    def fake(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    reload_schedule(user_config, config_path=cfg_path)

    run_plist = isolated_paths["launch_agents"] / "com.argos.run.plist"
    brief_plist = isolated_paths["launch_agents"] / "com.argos.brief.plist"
    run_data = plistlib.loads(run_plist.read_bytes())
    brief_data = plistlib.loads(brief_plist.read_bytes())

    # Both ProgramArguments must include ["--config", <cfg_path>] tail.
    assert run_data["ProgramArguments"][-2:] == ["--config", str(cfg_path)]
    assert brief_data["ProgramArguments"][-2:] == ["--config", str(cfg_path)]


def test_cli_schedule_install_forwards_config_path(
    monkeypatch,
    capsys,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """End-to-end: `argos schedule install --config <path>` must produce
    plists whose ProgramArguments embed that exact path.
    """
    from argos.cli import main

    # Minimal TOML the loader will accept; we mostly care that load() resolves
    # ``path`` to the value we passed on the CLI. Use a UserConfig.load stub
    # so we don't need a fully-valid config file.
    cfg_path = tmp_path / "user-config.toml"
    cfg_path.write_text("# stub\n")

    captured: dict[str, Path | None] = {"path": None}

    def fake_load(*, path):  # noqa: ANN001
        captured["path"] = path
        return SimpleNamespace(
            run=SimpleNamespace(time="06:00"),
            briefing=SimpleNamespace(
                time="07:00", weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"]
            ),
        )

    import argos.cli as cli_mod

    monkeypatch.setattr(cli_mod.UserConfig, "load", staticmethod(fake_load))

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "--config", str(cfg_path), "install"])
    assert rc == 0
    assert captured["path"] == cfg_path

    run_plist = isolated_paths["launch_agents"] / "com.argos.run.plist"
    brief_plist = isolated_paths["launch_agents"] / "com.argos.brief.plist"
    run_data = plistlib.loads(run_plist.read_bytes())
    brief_data = plistlib.loads(brief_plist.read_bytes())
    assert "--config" in run_data["ProgramArguments"]
    assert str(cfg_path) in run_data["ProgramArguments"]
    assert "--config" in brief_data["ProgramArguments"]
    assert str(cfg_path) in brief_data["ProgramArguments"]


# ---------------------------------------------------------------------------
# Regression: malformed run.time / briefing.time surfaces as SchedulerError,
# not an uncaught ValueError traceback.
# (Codex PR #41 — PRRT_kwDOR4m8Js6BC_SJ)
# ---------------------------------------------------------------------------


def test_reload_schedule_translates_bad_time_to_scheduler_error(
    monkeypatch,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
) -> None:
    user_config = SimpleNamespace(
        run=SimpleNamespace(time="25:00"),  # invalid hour
        briefing=SimpleNamespace(
            time="07:00", weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"]
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(["launchctl", *args], 0, "", ""),
    )
    with pytest.raises(SchedulerError, match="invalid time format"):
        reload_schedule(user_config)


def test_cli_schedule_install_bad_time_exits_cleanly(
    monkeypatch,
    capsys,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """A malformed run.time in config must produce a clean non-zero exit
    with a clear stderr message — no Python traceback.
    """
    from argos.cli import main

    cfg_path = tmp_path / "user-config.toml"
    cfg_path.write_text("# stub\n")

    def fake_load(*, path):  # noqa: ANN001, ARG001
        return SimpleNamespace(
            run=SimpleNamespace(time="25:00"),  # bad
            briefing=SimpleNamespace(
            time="07:00", weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"]
        ),
        )

    import argos.cli as cli_mod

    monkeypatch.setattr(cli_mod.UserConfig, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        scheduler,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(["launchctl", *args], 0, "", ""),
    )

    rc = main(["schedule", "--config", str(cfg_path), "install"])
    captured = capsys.readouterr()
    assert rc != 0
    # Clean error on stderr, no traceback noise.
    assert "Scheduler error" in captured.err
    assert "invalid time format" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# Regression: a relative `--config` path must be resolved to an absolute
# path before being embedded in the plist's ProgramArguments. launchd runs
# jobs with its own cwd, so a verbatim relative path would resolve to a
# different file (or nothing) at scheduled-run time.
# (Codex PR #41 — PRRT_kwDOR4m8Js6BDVFo)
# ---------------------------------------------------------------------------


def test_cli_schedule_install_resolves_relative_config_path(
    monkeypatch,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """`argos schedule install --config ./relative.toml` must embed the
    fully-resolved absolute path in both plists' ProgramArguments.
    """
    from argos.cli import main

    cfg_path = tmp_path / "user-config.toml"
    cfg_path.write_text("# stub\n")

    monkeypatch.chdir(tmp_path)

    captured: dict[str, Path | None] = {"path": None}

    def fake_load(*, path):  # noqa: ANN001
        captured["path"] = path
        return SimpleNamespace(
            run=SimpleNamespace(time="06:00"),
            briefing=SimpleNamespace(
                time="07:00", weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"]
            ),
        )

    import argos.cli as cli_mod

    monkeypatch.setattr(cli_mod.UserConfig, "load", staticmethod(fake_load))

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "--config", "./user-config.toml", "install"])
    assert rc == 0

    # Loader must have received an absolute path (not the relative form).
    assert captured["path"] is not None
    assert captured["path"].is_absolute()
    assert captured["path"] == cfg_path.resolve()

    run_plist = isolated_paths["launch_agents"] / "com.argos.run.plist"
    brief_plist = isolated_paths["launch_agents"] / "com.argos.brief.plist"
    run_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    brief_args = plistlib.loads(brief_plist.read_bytes())["ProgramArguments"]

    # ProgramArguments must carry the resolved absolute path, not "./..."
    assert run_args[-2:] == ["--config", str(cfg_path.resolve())]
    assert brief_args[-2:] == ["--config", str(cfg_path.resolve())]
    assert run_args[-1].startswith("/")
    assert brief_args[-1].startswith("/")
    assert "./user-config.toml" not in run_args
    assert "./user-config.toml" not in brief_args


def test_cli_schedule_install_missing_config_exits_cleanly(
    monkeypatch,
    capsys,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """`argos schedule install --config ./nonexistent.toml` must exit
    non-zero with a clean error (no traceback) instead of silently
    embedding a broken path in the plist.
    """
    from argos.cli import main

    monkeypatch.chdir(tmp_path)

    # If UserConfig.load is reached, the strict resolve failed silently.
    def fake_load(*, path):  # noqa: ANN001, ARG001
        raise AssertionError(
            "UserConfig.load must not be called when --config points at "
            "a nonexistent file"
        )

    import argos.cli as cli_mod

    monkeypatch.setattr(cli_mod.UserConfig, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        scheduler,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(["launchctl", *args], 0, "", ""),
    )

    rc = main(["schedule", "--config", "./nonexistent.toml", "install"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "Config file not found" in captured.err
    assert "Traceback" not in captured.err

    # No plists should have been written for the failed install.
    assert not (isolated_paths["launch_agents"] / "com.argos.run.plist").exists()
    assert not (isolated_paths["launch_agents"] / "com.argos.brief.plist").exists()


def test_cli_schedule_install_default_config_path_is_absolute(
    monkeypatch,
    fake_argos_binary: Path,
    fake_uid: int,
    isolated_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    """With no `--config` flag, the default ``~/.config/argos/config.toml``
    must still resolve to an absolute path and end up in the plist.
    """
    from argos.cli import main
    from argos import config_store

    # Point HOME at tmp_path so default_config_path() lands inside the sandbox.
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_home_cfg = tmp_path / ".config" / "argos" / "config.toml"
    fake_home_cfg.parent.mkdir(parents=True, exist_ok=True)
    fake_home_cfg.write_text("# stub\n")

    # Sanity: default_config_path now points at our sandboxed file.
    assert config_store.default_config_path() == fake_home_cfg

    captured: dict[str, Path | None] = {"path": None}

    def fake_load(*, path):  # noqa: ANN001
        captured["path"] = path
        return SimpleNamespace(
            run=SimpleNamespace(time="06:00"),
            briefing=SimpleNamespace(
                time="07:00", weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"]
            ),
        )

    import argos.cli as cli_mod

    monkeypatch.setattr(cli_mod.UserConfig, "load", staticmethod(fake_load))

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "install"])
    assert rc == 0
    assert captured["path"] is not None
    assert captured["path"].is_absolute()

    run_plist = isolated_paths["launch_agents"] / "com.argos.run.plist"
    run_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    assert "--config" in run_args
    cfg_arg = run_args[run_args.index("--config") + 1]
    assert cfg_arg.startswith("/")
    assert cfg_arg == str(fake_home_cfg.resolve())
