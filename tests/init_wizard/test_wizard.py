from __future__ import annotations

import textwrap

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
    # Avoid loading the real UserConfig — accept optional path kwarg.
    monkeypatch.setattr(wizard, "_user_config", lambda path=None: {})

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
    # Accept optional path kwarg to match the real _user_config signature.
    monkeypatch.setattr(wizard, "_user_config", lambda path=None: {})

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


# ---------------------------------------------------------------------------
# Regression: Finding 1 — rebuild() uses new POSTGRES_PORT after infra step
# ---------------------------------------------------------------------------

def test_rebuild_database_uses_new_port_from_env(tmp_path, monkeypatch):
    """After infra rewrites .env, _rebuild_database must pick up the new port.

    We write a .env with POSTGRES_PORT=9999, call _rebuild_database, then
    verify that the module-level engine URL reflects port 9999 — not whatever
    was loaded at module import time.  No real network call is made.
    """
    import argos.database as db_module

    env_file = tmp_path / ".env"
    env_file.write_text(
        textwrap.dedent("""\
            POSTGRES_USER=argos
            POSTGRES_PASSWORD=argos_dev_password
            POSTGRES_DB=argos
            POSTGRES_HOST=localhost
            POSTGRES_PORT=9999
        """)
    )

    wizard._rebuild_database(env_file)

    # The URL on the rebuilt engine must contain the NEW port.
    url_str = str(db_module.engine.url)
    assert ":9999/" in url_str, f"expected port 9999 in engine URL, got: {url_str}"


# ---------------------------------------------------------------------------
# Regression: Finding 2 — config_path is threaded through to the schedule step
# ---------------------------------------------------------------------------

def test_run_full_threads_config_path_to_schedule_step(tmp_path, monkeypatch):
    """run_full(config_path=...) must pass the custom config to the schedule step.

    We write a TOML with a distinctive briefing.time, then verify that the
    UserConfig received by run_schedule_step reflects that value — not the
    built-in default.
    """
    config_toml = tmp_path / "config.toml"
    config_toml.write_text(
        textwrap.dedent("""\
            [briefing]
            time = "13:37"
        """)
    )

    captured: dict = {}

    def fake_schedule_step(user_config):
        captured["user_config"] = user_config

    monkeypatch.setattr(wizard, "run_precheck_step", lambda: None)
    monkeypatch.setattr(wizard, "run_infra_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_slack_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_interests_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_schedule_step", fake_schedule_step)
    monkeypatch.setattr(wizard, "run_healthcheck_step", lambda *a, **kw: 0)

    rc = wizard.run_full(repo_root=tmp_path, config_path=config_toml)
    assert rc == 0

    assert "user_config" in captured, "run_schedule_step was never called"
    assert captured["user_config"].briefing.time == "13:37", (
        f"expected briefing.time='13:37', got {captured['user_config'].briefing.time!r}"
    )


# ---------------------------------------------------------------------------
# Regression: Finding 2 — run_reconfigure rebuilds DB engine before healthcheck
# for non-infra sections when a non-default env_path is supplied
# ---------------------------------------------------------------------------

def test_run_reconfigure_slack_rebuilds_db_engine_from_env_path(tmp_path, monkeypatch):
    """run_reconfigure('slack', env_path=tmp_env) must refresh the DB engine
    from tmp_env before the healthcheck, so db_ping targets the correct host.

    A non-default env_path with POSTGRES_PORT=9999 is supplied; after
    run_reconfigure returns we assert the module-level engine URL reflects
    port 9999.  No real DB or Slack calls are made.
    """
    import argos.database as db_module

    env_file = tmp_path / ".env"
    env_file.write_text(
        textwrap.dedent("""\
            POSTGRES_USER=argos
            POSTGRES_PASSWORD=argos_dev_password
            POSTGRES_DB=argos
            POSTGRES_HOST=localhost
            POSTGRES_PORT=9999
        """)
    )

    monkeypatch.setattr(wizard, "run_slack_step", lambda *a, **kw: None)
    monkeypatch.setattr(wizard, "run_healthcheck_step", lambda *a, **kw: 0)

    rc = wizard.run_reconfigure("slack", repo_root=tmp_path, env_path=env_file)
    assert rc == 0

    url_str = str(db_module.engine.url)
    assert ":9999/" in url_str, (
        f"expected port 9999 in engine URL after run_reconfigure('slack'), got: {url_str}"
    )
