"""ARG-332: 실제 LaunchAgents / launchctl 보호 장치(conftest ``_guard_real_launchd``) 검증.

프로브 파일은 존재하지 않는 이름이라, 가드가 실제 동작 전에 막으므로 파일 시스템은
바뀌지 않는다.
"""

from __future__ import annotations

import os
import plistlib
import subprocess

import pytest

from argos import scheduler
from tests.conftest import _REAL_LAUNCH_AGENTS

PROBE = _REAL_LAUNCH_AGENTS / "com.argos.__arg332_guard_probe__.plist"


def test_guard_blocks_unlink_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        PROBE.unlink(missing_ok=True)


@pytest.fixture
def _remove_probe_if_guard_regressed():
    """가드가 깨져 프로브가 실제로 만들어졌다면 운영 디렉터리에 남기지 않는다."""
    yield
    if PROBE.exists():
        PROBE.unlink()


@pytest.mark.usefixtures("_remove_probe_if_guard_regressed")
def test_guard_blocks_write_bytes_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        PROBE.write_bytes(b"x")
    assert not PROBE.exists()


@pytest.mark.usefixtures("_remove_probe_if_guard_regressed")
def test_guard_blocks_open_for_write_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        PROBE.open("wb")
    assert not PROBE.exists()


@pytest.mark.usefixtures("_remove_probe_if_guard_regressed")
def test_guard_blocks_builtin_open_for_write_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        open(PROBE, "w")
    assert not PROBE.exists()


def test_guard_blocks_os_replace_into_real_launch_agents(tmp_path):
    src = tmp_path / "staged.plist"
    src.write_text("x")
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        os.replace(src, PROBE)
    assert src.exists()


@pytest.mark.usefixtures("_remove_probe_if_guard_regressed")
def test_guard_blocks_install_plist_into_real_launch_agents():
    """실제 설치 경로(내장 open → os.replace)도 가드에 걸린다."""
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        scheduler.install_plist(PROBE, plistlib.dumps({"Label": "probe"}).decode("utf-8"))
    assert not PROBE.exists()
    assert not PROBE.with_suffix(".plist.tmp").exists()


def test_guard_blocks_os_remove_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        os.remove(PROBE)


def test_guard_blocks_real_launchctl():
    with pytest.raises(pytest.fail.Exception, match="launchctl"):
        scheduler._run_launchctl(["print", "gui/0/com.argos.run"])


def test_guard_allows_reads_on_real_launch_agents():
    # 읽기는 막지 않는다 — 예외 없이 통과하고, 실제 디렉터리 존재 여부를 그대로 읽는다
    assert PROBE.exists() is False
    assert PROBE.parent.exists() == os.path.isdir(_REAL_LAUNCH_AGENTS)


def test_guard_allows_writes_elsewhere(tmp_path):
    p = tmp_path / "com.argos.run.plist"
    p.write_bytes(b"x")
    p.unlink()


def test_uninstall_without_isolation_fails_loudly(monkeypatch):
    """원인 패턴 재현: isolated_paths 없이 uninstall → 보호 장치가 실패를 낸다."""
    from argos.cli import main

    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)
    monkeypatch.setattr(
        scheduler,
        "_run_launchctl",
        lambda args: subprocess.CompletedProcess(["launchctl", *args], 36, "", "not loaded"),
    )
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        main(["schedule", "uninstall"])
