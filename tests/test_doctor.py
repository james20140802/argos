"""Tests for argos.doctor — one test per probe, happy-path + failure cases."""

from __future__ import annotations

import os
import stat
import subprocess

import pytest

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
        lambda **kw: ["qwen3:8b", "qwen3:32b", "nomic-embed-text", "gemma4:26b"],
    )
    rows = doctor.check_ollama_models()
    assert len(rows) == 4
    assert all(status == "OK" for _, status, _ in rows)
    names = [name for name, _, _ in rows]
    assert "qwen3:8b" in names
    assert "qwen3:32b" in names
    assert "nomic-embed-text" in names
    assert "gemma4:26b" in names  # pair-check model (ARG-307)


def test_check_ollama_models_partial_missing(monkeypatch):
    """Only 8b present; 32b and embed missing → one OK, two FAIL rows."""
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list",
        lambda **kw: ["qwen3:8b"],
    )
    rows = doctor.check_ollama_models()
    assert len(rows) == 4
    by_name = {name: status for name, status, _ in rows}
    assert by_name["qwen3:8b"] == "OK"
    assert by_name["qwen3:32b"] == "FAIL"
    assert by_name["nomic-embed-text"] == "FAIL"
    assert by_name["gemma4:26b"] == "FAIL"


def test_check_ollama_models_ollama_unreachable(monkeypatch):
    """Ollama unreachable → all three rows FAIL with the same error."""
    def _raise(**kw):
        raise WizardStepError("could not reach Ollama at http://localhost:11434")

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _raise)
    rows = doctor.check_ollama_models()
    assert len(rows) == 4
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


@pytest.fixture(autouse=True)
def _stub_event_summary_probe(request, monkeypatch):
    """ARG-310: the CLI doctor tests must not reach a real Ollama through the
    event-summary probe. Tests of the probe itself (name contains
    ``check_event_summary_model``) keep the real function."""
    if "check_event_summary_model" in request.node.name:
        return
    monkeypatch.setattr(
        "argos.doctor.check_event_summary_model",
        lambda **kw: ("Event summary model", "OK", ""),
    )


def test_doctor_missing_event_summary_model_warns_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr("argos.doctor.check_docker", lambda: ("Docker daemon", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_ollama_installed", lambda: ("Ollama installed", "OK", ""))
    monkeypatch.setattr(
        "argos.doctor.check_ollama_models",
        lambda **kw: [("qwen3:8b", "OK", ""), ("qwen3:32b", "OK", ""), ("nomic-embed-text", "OK", "")],
    )
    monkeypatch.setattr(
        "argos.doctor.check_event_summary_model",
        lambda **kw: ("Event summary model (nope:1b)", "WARN", "model not found -- run: ollama pull nope:1b"),
    )
    monkeypatch.setattr("argos.doctor.check_python_version", lambda: ("Python version", "OK", "3.11.0"))
    monkeypatch.setattr("argos.doctor.check_macos_version", lambda: ("macOS version", "OK", "13.0.0"))
    monkeypatch.setattr("argos.doctor.check_uv_installed", lambda: ("uv installed", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_postgres_reachable", lambda: ("Postgres reachable", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_alembic_head", lambda: ("Alembic migrations", "OK", "abc123"))
    monkeypatch.setattr(
        "argos.doctor.check_vram_headroom", lambda **kw: ("VRAM headroom", "OK", "12.0 GiB free")
    )

    from argos.cli import main

    assert main(["doctor"]) == 0
    assert "ollama pull nope:1b" in capsys.readouterr().out


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
    monkeypatch.setattr("argos.doctor.check_postgres_reachable", lambda: ("Postgres reachable", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_alembic_head", lambda: ("Alembic migrations", "OK", "abc123"))
    monkeypatch.setattr(
        "argos.doctor.check_vram_headroom", lambda **kw: ("VRAM headroom", "OK", "12.0 GiB free")
    )

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
    monkeypatch.setattr("argos.doctor.check_postgres_reachable", lambda: ("Postgres reachable", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_alembic_head", lambda: ("Alembic migrations", "OK", "abc123"))
    monkeypatch.setattr(
        "argos.doctor.check_vram_headroom", lambda **kw: ("VRAM headroom", "OK", "12.0 GiB free")
    )

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
    monkeypatch.setattr("argos.doctor.check_postgres_reachable", lambda: ("Postgres reachable", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_alembic_head", lambda: ("Alembic migrations", "OK", "abc123"))
    # VRAM WARN too — proves multiple simultaneous WARNs still exit 0.
    monkeypatch.setattr(
        "argos.doctor.check_vram_headroom", lambda **kw: ("VRAM headroom", "WARN", "low headroom")
    )

    from argos.cli import main
    rc = main(["doctor"])
    assert rc == 0


def test_doctor_config_override_passes_custom_ollama_host(monkeypatch, tmp_path, capsys):
    """argos doctor --config <path> honours ollama.host from the config file."""
    cfg = tmp_path / "custom.toml"
    cfg.write_text('[ollama]\nhost = "http://custom-host:9999"\n')

    captured: dict = {}

    def _capture_models(**kw):
        captured.update(kw)
        return [("qwen3:8b", "OK", ""), ("qwen3:32b", "OK", ""), ("nomic-embed-text", "OK", "")]

    vram_captured: dict = {}

    def _capture_vram(**kw):
        vram_captured.update(kw)
        return ("VRAM headroom", "OK", "12.0 GiB free")

    monkeypatch.setattr("argos.doctor.check_docker", lambda: ("Docker daemon", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_ollama_installed", lambda: ("Ollama installed", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_ollama_models", _capture_models)
    monkeypatch.setattr("argos.doctor.check_python_version", lambda: ("Python version", "OK", "3.11.0"))
    monkeypatch.setattr("argos.doctor.check_macos_version", lambda: ("macOS version", "OK", "13.0.0"))
    monkeypatch.setattr("argos.doctor.check_uv_installed", lambda: ("uv installed", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_postgres_reachable", lambda: ("Postgres reachable", "OK", ""))
    monkeypatch.setattr("argos.doctor.check_alembic_head", lambda: ("Alembic migrations", "OK", "abc123"))
    monkeypatch.setattr("argos.doctor.check_vram_headroom", _capture_vram)

    from argos.cli import main
    rc = main(["doctor", "--config", str(cfg)])
    assert rc == 0
    assert captured.get("ollama_host") == "http://custom-host:9999"
    assert vram_captured.get("ollama_host") == "http://custom-host:9999"


# ---------------------------------------------------------------------------
# check_postgres_reachable
# ---------------------------------------------------------------------------


def test_check_postgres_reachable_ok(monkeypatch):
    # runners.run_async(db_ping()) 가 예외 없이 끝나면 OK.
    # Close the coroutine we were handed so pytest doesn't warn about an
    # un-awaited db_ping() (the real run_async awaits it).
    monkeypatch.setattr("argos.init_wizard.runners.run_async", lambda coro: coro.close())
    name, status, detail = doctor.check_postgres_reachable()
    assert name == "Postgres reachable"
    assert status == "OK"


def test_check_postgres_reachable_fail(monkeypatch):
    from argos.init_wizard import WizardStepError

    def _raise(coro):
        coro.close()  # avoid "coroutine was never awaited" warning
        raise WizardStepError("database ping failed: connection refused", hint="check db container")

    monkeypatch.setattr("argos.init_wizard.runners.run_async", _raise)
    name, status, detail = doctor.check_postgres_reachable()
    assert status == "FAIL"
    assert "database ping failed" in detail


# ---------------------------------------------------------------------------
# check_alembic_head
# ---------------------------------------------------------------------------


def test_check_alembic_head_up_to_date(monkeypatch):
    monkeypatch.setattr(
        "argos.doctor._alembic_current_and_head", lambda: ("abc123", "abc123")
    )
    name, status, detail = doctor.check_alembic_head()
    assert name == "Alembic migrations"
    assert status == "OK"


def test_check_alembic_head_behind(monkeypatch):
    monkeypatch.setattr(
        "argos.doctor._alembic_current_and_head", lambda: ("abc123", "def456")
    )
    _, status, detail = doctor.check_alembic_head()
    assert status == "FAIL"
    assert "upgrade head" in detail


def test_check_alembic_head_unreadable(monkeypatch):
    def _raise():
        raise RuntimeError("cannot connect")

    monkeypatch.setattr("argos.doctor._alembic_current_and_head", _raise)
    _, status, detail = doctor.check_alembic_head()
    assert status == "FAIL"
    assert "cannot connect" in detail


def test_check_alembic_head_installed_layout_is_soft_warn(monkeypatch):
    # Installed (pipx/wheel) layout: migration scripts aren't on disk. That is
    # not a DB fault, so the probe must WARN (non-fatal), not FAIL and force
    # `argos doctor` to exit non-zero on a healthy DB (P2, PR #113 review).
    def _raise():
        raise doctor._AlembicScriptsUnavailable("/opt/venv/lib/python3.12")

    monkeypatch.setattr("argos.doctor._alembic_current_and_head", _raise)
    _, status, detail = doctor.check_alembic_head()
    assert status == "WARN"
    assert "source checkout" in detail


# ---------------------------------------------------------------------------
# check_vram_headroom
# ---------------------------------------------------------------------------


def test_check_vram_headroom_ok(monkeypatch):
    monkeypatch.setattr("argos.doctor._available_memory_bytes", lambda: 12 * 1024**3)
    monkeypatch.setattr("argos.doctor._loaded_ollama_models", lambda host: ["qwen3:8b"])
    name, status, detail = doctor.check_vram_headroom()
    assert name == "VRAM headroom"
    assert status == "OK"
    assert "12" in detail  # free GiB surfaced


def test_check_vram_headroom_low_warns(monkeypatch):
    monkeypatch.setattr("argos.doctor._available_memory_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr("argos.doctor._loaded_ollama_models", lambda host: ["qwen3:32b"])
    _, status, detail = doctor.check_vram_headroom()
    assert status == "WARN"


def test_check_vram_headroom_ollama_unreachable_warns(monkeypatch):
    monkeypatch.setattr("argos.doctor._available_memory_bytes", lambda: 12 * 1024**3)

    def _raise(host):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("argos.doctor._loaded_ollama_models", _raise)
    _, status, _ = doctor.check_vram_headroom()
    assert status == "WARN"


def test_check_vram_headroom_memory_unknown_warns(monkeypatch):
    monkeypatch.setattr("argos.doctor._available_memory_bytes", lambda: None)
    monkeypatch.setattr("argos.doctor._loaded_ollama_models", lambda host: [])
    _, status, _ = doctor.check_vram_headroom()
    assert status == "WARN"


# ---------------------------------------------------------------------------
# _cmd_doctor registration smoke test
# ---------------------------------------------------------------------------


def test_doctor_module_exposes_new_probes():
    assert hasattr(doctor, "check_postgres_reachable")
    assert hasattr(doctor, "check_alembic_head")
    assert hasattr(doctor, "check_vram_headroom")
    for name in ("check_postgres_reachable", "check_alembic_head", "check_vram_headroom"):
        assert name in doctor.__all__


# ---------------------------------------------------------------------------
# check_event_summary_model (ARG-310)
# ---------------------------------------------------------------------------


def test_check_event_summary_model_present(monkeypatch):
    monkeypatch.setattr(
        "argos.init_wizard.runners.ollama_list", lambda **kw: ["gemma4:26b", "qwen3:8b"]
    )
    name, status, detail = doctor.check_event_summary_model(model="gemma4:26b")
    assert status == "OK"
    assert "gemma4:26b" in name


def test_check_event_summary_model_missing_is_a_warning_with_pull_hint(monkeypatch):
    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", lambda **kw: ["qwen3:8b"])
    _, status, detail = doctor.check_event_summary_model(model="gemma4:26b")
    assert status == "WARN"
    assert "ollama pull gemma4:26b" in detail


def test_check_event_summary_model_ollama_unreachable_is_a_warning(monkeypatch):
    def _raise(**kw):
        raise WizardStepError("could not reach Ollama at http://localhost:11434")

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _raise)
    _, status, detail = doctor.check_event_summary_model()
    assert status == "WARN"
    assert detail


def test_check_event_summary_model_forwards_host(monkeypatch):
    captured: dict = {}

    def _capture(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr("argos.init_wizard.runners.ollama_list", _capture)
    doctor.check_event_summary_model(ollama_host="http://example:1234")
    assert captured["host"] == "http://example:1234"


def test_event_summary_model_is_not_a_required_model():
    from argos.init_wizard.steps.infra import REQUIRED_OLLAMA_MODELS

    assert "gemma4:26b" not in REQUIRED_OLLAMA_MODELS


# ---------------------------------------------------------------------------
# ARG-195: secret file permission probe (read-only)
# ---------------------------------------------------------------------------


def _secret_paths(tmp_path):
    d = tmp_path / "argos"
    d.mkdir()
    os.chmod(d, 0o700)
    env = d / ".env"
    env.write_text("POSTGRES_PASSWORD=x\n")
    os.chmod(env, 0o600)
    tok = d / "web_token"
    tok.write_text("t\n")
    os.chmod(tok, 0o600)
    return d, env, tok


def test_secret_file_modes_ok_when_owner_only(tmp_path):
    from argos.doctor import check_secret_file_modes

    d, env, tok = _secret_paths(tmp_path)
    name, status, detail = check_secret_file_modes([d, env, tok, d / "missing"])
    assert status == "OK", detail


def test_secret_file_modes_warns_on_644_without_changing_it(tmp_path):
    from argos.doctor import check_secret_file_modes

    d, env, tok = _secret_paths(tmp_path)
    os.chmod(env, 0o644)
    os.chmod(d, 0o755)
    name, status, detail = check_secret_file_modes([d, env, tok])
    assert status == "WARN"
    assert f"chmod 600 {env}" in detail
    assert f"chmod 700 {d}" in detail
    assert str(tok) not in detail
    # read-only: modes untouched
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o644
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o755


def test_secret_file_modes_default_paths_cover_env_token_and_dir(tmp_path, monkeypatch):
    from argos.doctor import secret_file_paths

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("ARGOS_WEB_TOKEN_FILE", raising=False)
    monkeypatch.delenv("ARGOS_ENV_FILE", raising=False)
    paths = secret_file_paths()
    assert tmp_path / "argos" in paths
    assert tmp_path / "argos" / "web_token" in paths
