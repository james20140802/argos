"""Tests for argos.doctor — one test per probe, happy-path + failure cases."""

from __future__ import annotations

import subprocess

import argos.doctor as doctor
from argos.init_wizard import WizardStepError


# ---------------------------------------------------------------------------
# check_docker
# ---------------------------------------------------------------------------


def _make_proc(returncode=0, stdout="", stderr=""):
    class _P:
        pass
    p = _P()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_check_docker_ok(monkeypatch):
    """Docker binary found + daemon responsive → OK."""
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: "/usr/bin/docker")
    monkeypatch.setattr(
        "argos.doctor.subprocess.run",
        lambda *a, **kw: _make_proc(0),
    )
    name, status, detail = doctor.check_docker()
    assert name == "Docker daemon"
    assert status == "OK"
    assert detail == ""


def test_check_docker_binary_missing(monkeypatch):
    """Docker binary not on PATH → FAIL."""
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: None)
    _, status, detail = doctor.check_docker()
    assert status == "FAIL"
    assert "docker" in detail.lower()


def test_check_docker_daemon_down(monkeypatch):
    """Docker binary present but 'docker info' exits non-zero → FAIL."""
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: "/usr/bin/docker")
    monkeypatch.setattr(
        "argos.doctor.subprocess.run",
        lambda *a, **kw: _make_proc(1, stderr="Cannot connect to Docker daemon"),
    )
    _, status, detail = doctor.check_docker()
    assert status == "FAIL"
    assert detail  # non-empty error detail


def test_check_docker_daemon_timeout(monkeypatch):
    """docker info hangs → TimeoutExpired → FAIL."""
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: "/usr/bin/docker")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["docker", "info"], timeout=5)

    monkeypatch.setattr("argos.doctor.subprocess.run", _timeout)
    _, status, detail = doctor.check_docker()
    assert status == "FAIL"
    assert "timed out" in detail.lower()


# ---------------------------------------------------------------------------
# check_ollama_installed
# ---------------------------------------------------------------------------


def test_check_ollama_installed_ok(monkeypatch):
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: "/usr/local/bin/ollama")
    name, status, detail = doctor.check_ollama_installed()
    assert name == "Ollama installed"
    assert status == "OK"


def test_check_ollama_installed_missing(monkeypatch):
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: None)
    _, status, detail = doctor.check_ollama_installed()
    assert status == "FAIL"
    assert "ollama" in detail.lower()


# ---------------------------------------------------------------------------
# check_ollama_qwen3_8b
# ---------------------------------------------------------------------------


def test_check_ollama_qwen3_8b_present(monkeypatch):
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list",
        lambda **kw: ["qwen3:8b", "nomic-embed-text:latest"],
    )
    name, status, detail = doctor.check_ollama_qwen3_8b()
    assert name == "Qwen3-8B pulled"
    assert status == "OK"


def test_check_ollama_qwen3_8b_missing_from_list(monkeypatch):
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list",
        lambda **kw: ["nomic-embed-text:latest"],
    )
    _, status, detail = doctor.check_ollama_qwen3_8b()
    assert status == "FAIL"
    assert "qwen3:8b" in detail.lower()


def test_check_ollama_qwen3_8b_ollama_unreachable(monkeypatch):
    """ollama_list raises WizardStepError (Ollama not running) → FAIL."""
    def _raise(**kw):
        raise WizardStepError("could not reach Ollama at http://localhost:11434")

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _raise)
    _, status, detail = doctor.check_ollama_qwen3_8b()
    assert status == "FAIL"
    assert detail  # non-empty


def test_check_ollama_qwen3_8b_uses_configured_host(monkeypatch):
    """ollama_host kwarg is forwarded to runners.ollama_list."""
    captured: dict = {}

    def _capture(**kw):
        captured.update(kw)
        return ["qwen3:8b"]

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _capture)
    doctor.check_ollama_qwen3_8b(ollama_host="http://custom-host:12345")
    assert captured.get("host") == "http://custom-host:12345"


# ---------------------------------------------------------------------------
# check_ollama_models
# ---------------------------------------------------------------------------


def test_check_ollama_models_all_present(monkeypatch):
    """All three required models present (exact IDs) → three OK rows."""
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list",
        lambda **kw: ["qwen3:8b", "qwen3:32b", "nomic-embed-text"],
    )
    rows = doctor.check_ollama_models()
    assert len(rows) == 3
    assert all(status == "OK" for _, status, _ in rows)
    names = [name for name, _, _ in rows]
    assert "qwen3:8b" in names
    assert "qwen3:32b" in names
    assert "nomic-embed-text" in names


def test_check_ollama_models_partial_missing(monkeypatch):
    """Only 8b present; 32b and embed missing → one OK, two FAIL rows."""
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list",
        lambda **kw: ["qwen3:8b"],
    )
    rows = doctor.check_ollama_models()
    assert len(rows) == 3
    by_name = {name: status for name, status, _ in rows}
    assert by_name["qwen3:8b"] == "OK"
    assert by_name["qwen3:32b"] == "FAIL"
    assert by_name["nomic-embed-text"] == "FAIL"


def test_check_ollama_models_ollama_unreachable(monkeypatch):
    """Ollama unreachable → all three rows FAIL with the same error."""
    def _raise(**kw):
        raise WizardStepError("could not reach Ollama at http://localhost:11434")

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _raise)
    rows = doctor.check_ollama_models()
    assert len(rows) == 3
    assert all(status == "FAIL" for _, status, _ in rows)
    assert all(detail for _, _, detail in rows)


def test_check_ollama_models_tagged_variant_does_not_satisfy_exact_name(monkeypatch):
    """A differently-tagged variant (e.g. qwen3:8b-instruct) must NOT satisfy
    the requirement for the exact ID qwen3:8b — doctor should report FAIL so
    the user knows to pull the correct tag."""
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list",
        lambda **kw: ["qwen3:8b-instruct", "qwen3:32b", "nomic-embed-text"],
    )
    rows = doctor.check_ollama_models()
    by_name = {name: status for name, status, _ in rows}
    assert by_name["qwen3:8b"] == "FAIL"
    assert by_name["qwen3:32b"] == "OK"
    assert by_name["nomic-embed-text"] == "OK"


def test_check_ollama_models_uses_configured_host(monkeypatch):
    """ollama_host kwarg is forwarded to runners.ollama_list."""
    captured: dict = {}

    def _capture(**kw):
        captured.update(kw)
        return ["qwen3:8b", "qwen3:32b", "nomic-embed-text"]

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _capture)
    doctor.check_ollama_models(ollama_host="http://custom-host:12345")
    assert captured.get("host") == "http://custom-host:12345"


# ---------------------------------------------------------------------------
# check_uv_installed
# ---------------------------------------------------------------------------


def test_check_uv_installed_ok(monkeypatch):
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: "/usr/local/bin/uv")
    name, status, detail = doctor.check_uv_installed()
    assert name == "uv installed"
    assert status == "OK"
    assert detail == ""


def test_check_uv_installed_missing(monkeypatch):
    monkeypatch.setattr("argos.init_wizard.runners.which", lambda b: None)
    _, status, detail = doctor.check_uv_installed()
    assert status == "FAIL"
    assert "uv" in detail.lower()


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------


def test_check_python_version_ok(monkeypatch):
    """Current interpreter is 3.11 → OK."""
    monkeypatch.setattr("argos.doctor.sys.version_info", (3, 11, 2, "final", 0))
    name, status, detail = doctor.check_python_version()
    assert name == "Python version"
    assert status == "OK"
    assert "3.11" in detail


def test_check_python_version_too_old(monkeypatch):
    monkeypatch.setattr("argos.doctor.sys.version_info", (3, 9, 0, "final", 0))
    _, status, detail = doctor.check_python_version()
    assert status == "FAIL"
    assert "3.9" in detail


def test_check_python_version_too_new(monkeypatch):
    monkeypatch.setattr("argos.doctor.sys.version_info", (3, 13, 0, "final", 0))
    _, status, detail = doctor.check_python_version()
    assert status == "FAIL"
    assert "3.13" in detail


def test_check_python_version_lower_boundary(monkeypatch):
    """3.10 is exactly supported (>=3.10) → OK."""
    monkeypatch.setattr("argos.doctor.sys.version_info", (3, 10, 0, "final", 0))
    _, status, _ = doctor.check_python_version()
    assert status == "OK"


def test_check_python_version_upper_boundary(monkeypatch):
    """3.12 is the last fully supported minor → OK."""
    monkeypatch.setattr("argos.doctor.sys.version_info", (3, 12, 5, "final", 0))
    _, status, _ = doctor.check_python_version()
    assert status == "OK"


# ---------------------------------------------------------------------------
# check_macos_version
# ---------------------------------------------------------------------------


def test_check_macos_version_ok(monkeypatch):
    """macOS 13 (Ventura) → OK."""
    monkeypatch.setattr("argos.doctor.platform.mac_ver", lambda: ("13.5.0", ("", "", ""), "arm64"))
    name, status, detail = doctor.check_macos_version()
    assert name == "macOS version"
    assert status == "OK"
    assert "13.5.0" in detail


def test_check_macos_version_minimum(monkeypatch):
    """macOS 12 (Monterey) exactly → OK."""
    monkeypatch.setattr("argos.doctor.platform.mac_ver", lambda: ("12.0.0", ("", "", ""), "arm64"))
    _, status, _ = doctor.check_macos_version()
    assert status == "OK"


def test_check_macos_version_too_old_is_warn(monkeypatch):
    """macOS 11 → WARN (not FAIL), so it doesn't block exit code."""
    monkeypatch.setattr("argos.doctor.platform.mac_ver", lambda: ("11.7.0", ("", "", ""), "arm64"))
    _, status, detail = doctor.check_macos_version()
    assert status == "WARN"
    assert "11" in detail


def test_check_macos_version_non_macos(monkeypatch):
    """Non-macOS host (mac_ver returns empty string) → OK with skip note."""
    monkeypatch.setattr("argos.doctor.platform.mac_ver", lambda: ("", ("", "", ""), ""))
    _, status, detail = doctor.check_macos_version()
    assert status == "OK"
    assert "skip" in detail.lower() or "macOS check skipped" in detail


# ---------------------------------------------------------------------------
# Integration: cli.main(["doctor"]) exit code matches failure count
# ---------------------------------------------------------------------------


def test_doctor_command_exits_zero_when_all_ok(monkeypatch, capsys):
    """When every probe passes, `argos doctor` returns 0."""
    # Patch the individual probe functions to return OK rows directly, avoiding
    # the need to monkey-patch sys.version_info globally (which breaks bs4 etc.).
    monkeypatch.setattr("argos.doctor.check_docker", lambda: ("Docker daemon", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_ollama_installed", lambda: ("Ollama installed", "OK", ""))
    monkeypatch.setattr(
        "argos.doctor.check_ollama_models",
        lambda **kw: [("qwen3:8b", "OK", ""), ("qwen3:32b", "OK", ""), ("nomic-embed-text", "OK", "")],
    )
    monkeypatch.setattr("argos.doctor.check_python_version", lambda: ("Python version", "OK", "3.11.0"))
    monkeypatch.setattr("argos.doctor.check_macos_version", lambda: ("macOS version", "OK", "13.0.0"))
    monkeypatch.setattr("argos.doctor.check_uv_installed", lambda: ("uv installed", "OK", ""))

    from argos.cli import main
    rc = main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "argos doctor" in out


def test_doctor_command_exits_nonzero_when_probe_fails(monkeypatch, capsys):
    """When at least one probe FAILs, `argos doctor` returns non-zero."""
    monkeypatch.setattr("argos.doctor.check_docker", lambda: ("Docker daemon", "FAIL", "daemon not running"))
    monkeypatch.setattr("argos.doctor.check_ollama_installed", lambda: ("Ollama installed", "FAIL", "not found"))
    monkeypatch.setattr(
        "argos.doctor.check_ollama_models",
        lambda **kw: [
            ("qwen3:8b", "FAIL", "not pulled"),
            ("qwen3:32b", "FAIL", "not pulled"),
            ("nomic-embed-text", "FAIL", "not pulled"),
        ],
    )
    monkeypatch.setattr("argos.doctor.check_python_version", lambda: ("Python version", "OK", "3.11.0"))
    monkeypatch.setattr("argos.doctor.check_macos_version", lambda: ("macOS version", "OK", "13.0.0"))
    monkeypatch.setattr("argos.doctor.check_uv_installed", lambda: ("uv installed", "OK", ""))

    from argos.cli import main
    rc = main(["doctor"])
    assert rc != 0


def test_doctor_warn_only_does_not_fail(monkeypatch, capsys):
    """macOS too-old is WARN, not FAIL → exit 0 when that's the only issue."""
    monkeypatch.setattr("argos.doctor.check_docker", lambda: ("Docker daemon", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_ollama_installed", lambda: ("Ollama installed", "OK", ""))
    monkeypatch.setattr(
        "argos.doctor.check_ollama_models",
        lambda **kw: [("qwen3:8b", "OK", ""), ("qwen3:32b", "OK", ""), ("nomic-embed-text", "OK", "")],
    )
    monkeypatch.setattr("argos.doctor.check_python_version", lambda: ("Python version", "OK", "3.11.0"))
    # macOS 11 → WARN only
    monkeypatch.setattr("argos.doctor.check_macos_version", lambda: ("macOS version", "WARN", "11.0.0 — old"))
    monkeypatch.setattr("argos.doctor.check_uv_installed", lambda: ("uv installed", "OK", ""))

    from argos.cli import main
    rc = main(["doctor"])
    assert rc == 0
