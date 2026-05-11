from __future__ import annotations

import pytest

from argos.init_wizard import WizardAbort
from argos.init_wizard.env_file import load_env
from argos.init_wizard.steps import slack as slack_step


@pytest.fixture(autouse=True)
def _force_noninteractive(monkeypatch):
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")


def _seed_env(env_path):
    env_path.write_text(
        "SLACK_BOT_TOKEN=xoxb-existing\n"
        "SLACK_APP_TOKEN=xapp-existing\n"
    )


def _seed_config(config_path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text('[slack]\nchannel_id = "C111"\nsummary_language = "Korean"\n')


def test_slack_step_validates_and_persists(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    _seed_config(cfg_path)

    seen_tokens = []

    def fake_auth(token, app_token=None):
        seen_tokens.append(token)
        return {"ok": True}

    monkeypatch.setattr(slack_step.runners, "slack_auth_test", fake_auth)

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)

    # Non-interactive mode reuses the existing values; auth was tested once.
    assert seen_tokens == ["xoxb-existing"]
    data = load_env(env_path)
    assert data["SLACK_BOT_TOKEN"] == "xoxb-existing"
    assert data["SLACK_APP_TOKEN"] == "xapp-existing"


def test_slack_step_loops_on_invalid_token_then_aborts(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("SLACK_BOT_TOKEN=not-a-token\nSLACK_APP_TOKEN=xapp-x\n")
    cfg_path = tmp_path / "config.toml"
    _seed_config(cfg_path)

    # Validator should immediately reject because the token doesn't start with xoxb-.
    with pytest.raises(WizardAbort):
        slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)


def test_slack_step_idempotent_when_channel_unchanged(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _seed_env(env_path)
    cfg_path = tmp_path / "config.toml"
    _seed_config(cfg_path)

    monkeypatch.setattr(slack_step.runners, "slack_auth_test", lambda t, a=None: {"ok": True})

    write_calls = []
    original_set = slack_step.config_store.set_value

    def spy_set(path, key, value):
        write_calls.append((key, value))
        return original_set(path, key, value)

    monkeypatch.setattr(slack_step.config_store, "set_value", spy_set)

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
    # Channel ID was already C111 — must not write.
    assert write_calls == []


def test_slack_step_writes_new_channel_id(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _seed_env(env_path)
    cfg_path = tmp_path / "config.toml"
    # Seed with empty channel so the non-interactive default ("") doesn't match.
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = "C111"\nsummary_language = "Korean"\n')

    monkeypatch.setattr(slack_step.runners, "slack_auth_test", lambda t, a=None: {"ok": True})

    # Patch ask_text to return a different channel id (simulating real input).
    monkeypatch.setattr(slack_step.prompts, "ask_text", lambda msg, default=None: "C222")
    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
    assert ("slack.channel_id", "C222") in write_calls
