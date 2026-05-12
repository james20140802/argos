from __future__ import annotations

import pytest

from argos.init_wizard import WizardAbort
from argos.init_wizard.env_file import file_mode, load_env
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
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: {"ok": True, "url": "wss://example.com"},
    )

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
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: {"ok": True, "url": "wss://example.com"},
    )

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
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: {"ok": True, "url": "wss://example.com"},
    )

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


# --- Regression test: Finding 2 (P2) — harden .env mode on no-op persist ---

def test_persist_tokens_hardens_env_mode_on_noop(tmp_path, monkeypatch):
    """When tokens are unchanged (early-return path), .env must be hardened to 0600."""
    env_path = tmp_path / ".env"
    env_path.write_text("SLACK_BOT_TOKEN=xoxb-existing\nSLACK_APP_TOKEN=xapp-existing\n")
    # Start with loose permissions to verify harden_env_file_mode fires.
    env_path.chmod(0o644)
    assert file_mode(env_path) == 0o644

    cfg_path = tmp_path / "config.toml"
    _seed_config(cfg_path)

    monkeypatch.setattr(slack_step.runners, "slack_auth_test", lambda t, a=None: {"ok": True})
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: {"ok": True, "url": "wss://example.com"},
    )

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)

    # No tokens changed → early-return path → harden_env_file_mode must enforce 0600.
    assert file_mode(env_path) == 0o600


# --- Regression tests for P1: channel ID validation ---

def _make_auth_stubs(monkeypatch):
    monkeypatch.setattr(slack_step.runners, "slack_auth_test", lambda t, a=None: {"ok": True})
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: {"ok": True, "url": "wss://example.com"},
    )


def test_channel_empty_input_aborts_on_fresh_install(tmp_path, monkeypatch):
    """Fresh install: current_channel="" + user presses Enter → WizardAbort, never persists empty."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = ""\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    # Noninteractive mode → ask_text always returns "" → all 3 attempts invalid.
    with pytest.raises(WizardAbort):
        slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)

    # Must never have written an empty channel ID.
    empty_writes = [(k, v) for k, v in write_calls if k == "slack.channel_id" and not v]
    assert empty_writes == []


def test_channel_existing_value_kept_on_enter(tmp_path, monkeypatch):
    """current_channel='C01234567' + user presses Enter → keeps existing value (real default)."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = "C01234567"\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    # Noninteractive mode + default="C01234567" → ask_text returns "C01234567".
    # Validator accepts; no write since value is unchanged.
    write_calls = []
    original_set = slack_step.config_store.set_value

    def spy_set(path, key, value):
        write_calls.append((key, value))
        return original_set(path, key, value)

    monkeypatch.setattr(slack_step.config_store, "set_value", spy_set)

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
    # Value unchanged → no write.
    assert ("slack.channel_id", "C01234567") not in write_calls


def test_channel_bogus_hash_general_aborts(tmp_path, monkeypatch):
    """Input '#general' (not a channel ID) → validator rejects every attempt → WizardAbort."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = ""\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    monkeypatch.setattr(slack_step.prompts, "ask_text", lambda msg, default=None: "#general")

    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    with pytest.raises(WizardAbort):
        slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)

    bogus_writes = [(k, v) for k, v in write_calls if k == "slack.channel_id"]
    assert bogus_writes == []


def test_channel_valid_id_persists(tmp_path, monkeypatch):
    """Valid input 'C09876543' → persisted to config.toml."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = ""\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    monkeypatch.setattr(slack_step.prompts, "ask_text", lambda msg, default=None: "C09876543")

    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
    assert ("slack.channel_id", "C09876543") in write_calls


def test_channel_g_prefix_accepted(tmp_path, monkeypatch):
    """G-prefix private channel ID is accepted and persisted to config.toml."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = ""\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    monkeypatch.setattr(slack_step.prompts, "ask_text", lambda msg, default=None: "G09876543")

    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
    assert ("slack.channel_id", "G09876543") in write_calls


def test_channel_c_prefix_still_accepted(tmp_path, monkeypatch):
    """Regression: C-prefix public channel ID continues to be accepted (ARG-71)."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = ""\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    monkeypatch.setattr(slack_step.prompts, "ask_text", lambda msg, default=None: "C01234567")

    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
    assert ("slack.channel_id", "C01234567") in write_calls


def test_channel_d_prefix_rejected(tmp_path, monkeypatch):
    """D-prefix DM channel ID is rejected → WizardAbort after 3 attempts, no write."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    _seed_env(env_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('[slack]\nchannel_id = ""\nsummary_language = "Korean"\n')

    _make_auth_stubs(monkeypatch)

    monkeypatch.setattr(slack_step.prompts, "ask_text", lambda msg, default=None: "D01234567")

    write_calls = []
    monkeypatch.setattr(
        slack_step.config_store,
        "set_value",
        lambda p, k, v: write_calls.append((k, v)),
    )

    with pytest.raises(WizardAbort):
        slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)

    dm_writes = [(k, v) for k, v in write_calls if k == "slack.channel_id"]
    assert dm_writes == []


# --- Regression tests for Finding 2: apps.connections.open validation ---

def test_app_token_validator_accepts_valid_xapp_token(tmp_path, monkeypatch):
    """apps.connections.open returns ok:true → validator accepts the token."""
    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    env_path.write_text("SLACK_BOT_TOKEN=xoxb-valid\nSLACK_APP_TOKEN=xapp-valid\n")
    _seed_config(cfg_path)

    monkeypatch.setattr(slack_step.runners, "slack_auth_test", lambda t, a=None: {"ok": True})
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: {"ok": True, "url": "wss://wss-primary.slack.com/link"},
    )

    # Should not raise — both tokens accepted.
    slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)


def test_app_token_validator_rejects_on_connections_open_failure(tmp_path, monkeypatch):
    """apps.connections.open returns ok:false → validator rejects, loop aborts after 3."""
    from argos.init_wizard import WizardStepError

    env_path = tmp_path / ".env"
    cfg_path = tmp_path / "config.toml"
    env_path.write_text("SLACK_BOT_TOKEN=xoxb-valid\nSLACK_APP_TOKEN=xapp-revoked\n")
    _seed_config(cfg_path)

    monkeypatch.setattr(slack_step.runners, "slack_auth_test", lambda t, a=None: {"ok": True})
    monkeypatch.setattr(
        slack_step.runners, "slack_app_connections_open",
        lambda t: (_ for _ in ()).throw(
            WizardStepError("slack app token rejected by apps.connections.open", hint="invalid_auth")
        ),
    )

    # Validation loop exhausts 3 attempts and raises WizardAbort.
    with pytest.raises(WizardAbort):
        slack_step.run_slack_step(tmp_path, env_path=env_path, config_path=cfg_path)
