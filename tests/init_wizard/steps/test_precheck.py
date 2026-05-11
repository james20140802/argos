from __future__ import annotations

import pytest

from argos.init_wizard import WizardAbort
from argos.init_wizard.steps import precheck


def test_precheck_passes_when_all_binaries_present(monkeypatch):
    monkeypatch.setattr(precheck.runners, "which", lambda b: f"/fake/{b}")
    # Should return None without raising.
    assert precheck.run_precheck_step() is None


def test_precheck_raises_abort_when_docker_missing(monkeypatch):
    presence = {"docker": None, "ollama": "/usr/local/bin/ollama", "uv": "/usr/local/bin/uv"}
    monkeypatch.setattr(precheck.runners, "which", lambda b: presence.get(b))
    with pytest.raises(WizardAbort) as excinfo:
        precheck.run_precheck_step()
    assert "docker" in str(excinfo.value)


def test_precheck_raises_abort_when_uv_missing(monkeypatch):
    presence = {"docker": "/usr/local/bin/docker", "ollama": "/usr/local/bin/ollama", "uv": None}
    monkeypatch.setattr(precheck.runners, "which", lambda b: presence.get(b))
    with pytest.raises(WizardAbort) as excinfo:
        precheck.run_precheck_step()
    msg = str(excinfo.value)
    assert "uv" in msg
    assert "https://github.com/astral-sh/uv" in msg


def test_precheck_lists_all_missing_binaries(monkeypatch):
    monkeypatch.setattr(precheck.runners, "which", lambda b: None)
    with pytest.raises(WizardAbort) as excinfo:
        precheck.run_precheck_step()
    msg = str(excinfo.value)
    assert "docker" in msg
    assert "ollama" in msg
    assert "uv" in msg
