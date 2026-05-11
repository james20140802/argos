from __future__ import annotations

import pytest

from argos.init_wizard import WizardAbort, WizardStepError, wizard


@pytest.fixture(autouse=True)
def _noninteractive(monkeypatch):
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")


def test_run_full_invokes_all_six_steps_in_order(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(wizard, "run_precheck_step", lambda: order.append("precheck"))
    monkeypatch.setattr(
        wizard,
        "run_infra_step",
        lambda repo, env_path=None: order.append("infra"),
    )
    monkeypatch.setattr(
        wizard,
        "run_slack_step",
        lambda repo, env_path=None, config_path=None: order.append("slack"),
    )
    monkeypatch.setattr(
        wizard,
        "run_interests_step",
        lambda config_path=None: order.append("interests"),
    )
    monkeypatch.setattr(wizard, "run_schedule_step", lambda cfg: order.append("schedule"))
    monkeypatch.setattr(
        wizard,
        "run_healthcheck_step",
        lambda repo, env_path=None: (order.append("healthcheck"), 0)[1],
    )
    # Avoid loading the real UserConfig.
    monkeypatch.setattr(wizard, "_user_config", lambda: {})

    rc = wizard.run_full(repo_root=tmp_path, env_path=tmp_path / ".env", config_path=tmp_path / "config.toml")

    assert rc == 0
    assert order == ["precheck", "infra", "slack", "interests", "schedule", "healthcheck"]


def test_run_full_returns_one_when_healthcheck_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard, "run_precheck_step", lambda: None)
    monkeypatch.setattr(wizard, "run_infra_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_slack_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_interests_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_schedule_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_healthcheck_step", lambda *a, **kw: 2)
    monkeypatch.setattr(wizard, "_user_config", lambda: {})

    rc = wizard.run_full(repo_root=tmp_path)
    assert rc == 1


def test_run_full_handles_wizard_abort_with_zero_exit(tmp_path, monkeypatch, capsys):
    def boom():
        raise WizardAbort("docker missing")

    monkeypatch.setattr(wizard, "run_precheck_step", boom)
    rc = wizard.run_full(repo_root=tmp_path)
    assert rc == 0
    assert "docker missing" in capsys.readouterr().out


def test_run_full_handles_wizard_step_error_with_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(wizard, "run_precheck_step", lambda: None)

    def boom(*a, **kw):
        raise WizardStepError("docker exited 1", hint="run docker info")

    monkeypatch.setattr(wizard, "run_infra_step", boom)
    rc = wizard.run_full(repo_root=tmp_path)
    assert rc == 1
    out = capsys.readouterr().out
    assert "error: docker exited 1" in out
    assert "hint: run docker info" in out


def test_run_reconfigure_dispatches_to_single_section(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(
        wizard,
        "run_interests_step",
        lambda config_path=None: order.append("interests"),
    )
    monkeypatch.setattr(
        wizard,
        "run_healthcheck_step",
        lambda repo, env_path=None: (order.append("healthcheck"), 0)[1],
    )

    rc = wizard.run_reconfigure("interests", repo_root=tmp_path)
    assert rc == 0
    assert order == ["interests", "healthcheck"]


def test_run_reconfigure_rejects_unknown_section(tmp_path):
    with pytest.raises(ValueError):
        wizard.run_reconfigure("nope", repo_root=tmp_path)
