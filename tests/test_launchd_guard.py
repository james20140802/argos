"""ARG-332: 실제 LaunchAgents / launchctl 보호 장치(conftest ``_guard_real_launchd``) 검증.

프로브 파일은 존재하지 않는 이름이라, 가드가 실제 동작 전에 막으므로 파일 시스템은
바뀌지 않는다.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from argos import scheduler
from tests.conftest import _REAL_LAUNCH_AGENTS

PROBE = _REAL_LAUNCH_AGENTS / "com.argos.__arg332_guard_probe__.plist"


def test_guard_blocks_unlink_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        PROBE.unlink(missing_ok=True)


def test_guard_blocks_write_bytes_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        PROBE.write_bytes(b"x")
    assert not PROBE.exists()


def test_guard_blocks_open_for_write_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        PROBE.open("wb")
    assert not PROBE.exists()


def test_guard_blocks_os_remove_on_real_launch_agents():
    with pytest.raises(pytest.fail.Exception, match="LaunchAgents"):
        os.remove(PROBE)


def test_guard_blocks_real_launchctl():
    with pytest.raises(pytest.fail.Exception, match="launchctl"):
        scheduler._run_launchctl(["print", "gui/0/com.argos.run"])


def test_guard_allows_reads_on_real_launch_agents():
    PROBE.exists()  # 읽기는 막지 않는다 — 예외 없이 통과


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
