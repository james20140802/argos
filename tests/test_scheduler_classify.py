"""하루 한 번 자동 갈래 배정 launchd job (ARG-313) — launchctl은 가짜."""
from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

import pytest

from argos import scheduler
from argos.scheduler import SchedulerError, reload_schedule, render_classify_plist


@pytest.fixture
def fake_argos_binary(monkeypatch, tmp_path) -> Path:
    binary = tmp_path / "argos"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(scheduler.shutil, "which", lambda name: str(binary))
    return binary.resolve()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)
    la = tmp_path / "LaunchAgents"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", tmp_path / "Logs")
    return la


def _launchctl(monkeypatch):
    calls: list[list[str]] = []

    def fake(args):
        calls.append(args)
        out = f"{args[1].rsplit('/', 1)[-1]} = service" if args[0] == "print" else ""
        return subprocess.CompletedProcess(["launchctl", *args], 0, stdout=out, stderr="")

    monkeypatch.setattr(scheduler, "_run_launchctl", fake)
    return calls


def _cfg(**taxonomy_kw):
    from argos.config import TaxonomyConfig, UserConfig

    cfg = UserConfig()
    cfg.run.time = "06:00"
    cfg.briefing.time = "07:00"
    cfg.taxonomy = TaxonomyConfig(**taxonomy_kw)
    return cfg


def test_render_classify_plist_runs_assign_yes_daily(fake_argos_binary, tmp_path):
    payload = plistlib.loads(render_classify_plist(time="04:30").encode())
    assert payload["Label"] == "com.argos.classify"
    assert payload["ProgramArguments"][1:] == ["taxonomy", "assign", "--yes"]
    assert payload["StartCalendarInterval"] == {"Hour": 4, "Minute": 30}
    assert payload["StandardOutPath"].endswith("classify.log")

    cfg = tmp_path / "c.toml"
    payload = plistlib.loads(render_classify_plist(time="04:30", config_path=cfg).encode())
    # --config는 taxonomy 파서가 받으므로 assign 앞에 온다.
    assert payload["ProgramArguments"][1:] == ["taxonomy", "--config", str(cfg), "assign", "--yes"]


def test_scheduled_command_parses(monkeypatch, fake_argos_binary, tmp_path):
    from argos import cli, taxonomy_cli

    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    argv = plistlib.loads(render_classify_plist(time="04:30", config_path=cfg).encode())[
        "ProgramArguments"
    ][1:]
    seen = {}

    def fake_run(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(taxonomy_cli, "run", fake_run)
    monkeypatch.setattr(cli, "_apply_config_override", lambda args: None)
    assert cli.main(argv) == 0
    ns = seen["args"]
    assert ns.taxonomy_action == "assign" and ns.yes and ns.config == str(cfg)


def test_reload_installs_classify_by_default(monkeypatch, fake_argos_binary, isolated):
    calls = _launchctl(monkeypatch)
    reload_schedule(_cfg())
    assert (isolated / "com.argos.classify.plist").exists()
    boots = [Path(a[2]).name for a in calls if a[0] == "bootstrap"]
    assert boots[-1] == "com.argos.classify.plist"


@pytest.mark.parametrize("clash", ["06:00", "07:00", "03:00"])
def test_reload_rejects_overlap_before_touching_anything(monkeypatch, fake_argos_binary, isolated, clash):
    calls = _launchctl(monkeypatch)
    with pytest.raises(SchedulerError, match="taxonomy.assign_time"):
        reload_schedule(_cfg(assign_time=clash))
    assert calls == []
    assert not isolated.exists() or list(isolated.iterdir()) == []


def test_disabled_boots_out_and_deletes_plist(monkeypatch, fake_argos_binary, isolated):
    isolated.mkdir(parents=True)
    stale = isolated / "com.argos.classify.plist"
    stale.write_text("<x/>")
    calls = _launchctl(monkeypatch)
    reload_schedule(_cfg(assign_enabled=False))
    assert not stale.exists()
    assert any(a[0] == "bootout" and a[1].endswith("com.argos.classify") for a in calls)


def test_cli_status_and_uninstall_include_classify(monkeypatch, capsys, isolated):
    from argos.cli import main

    _launchctl(monkeypatch)
    isolated.mkdir(parents=True)
    (isolated / "com.argos.classify.plist").write_text("<x/>")
    assert main(["schedule", "uninstall"]) == 0
    assert "Unloaded: com.argos.classify" in capsys.readouterr().out
    assert not (isolated / "com.argos.classify.plist").exists()
    main(["schedule", "status"])
    assert "com.argos.classify:" in capsys.readouterr().out
