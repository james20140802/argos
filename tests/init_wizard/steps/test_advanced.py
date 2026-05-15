from __future__ import annotations

import pytest

from argos.init_wizard.steps import advanced


@pytest.fixture(autouse=True)
def _force_noninteractive(monkeypatch):
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")


def test_advanced_step_skips_when_user_declines(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[triage]\npreflight_filter = true\n"
        "[briefing]\ntime = \"07:00\"\nweekdays = [\"Mon\"]\nlimit_per_category = 10\n"
        "[run]\ndaily_limit = 150\ntime = \"06:00\"\n"
    )
    monkeypatch.setattr(advanced.prompts, "ask_confirm", lambda *a, **kw: False)

    write_calls = []
    monkeypatch.setattr(
        advanced.config_store, "set_value", lambda p, k, v: write_calls.append(k)
    )

    advanced.run_advanced_step(config_path=cfg)

    assert write_calls == []


def test_advanced_step_writes_all_three_keys_when_changed(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[triage]\npreflight_filter = true\n"
        "[briefing]\ntime = \"07:00\"\nweekdays = [\"Mon\"]\nlimit_per_category = 10\n"
        "[run]\ndaily_limit = 150\ntime = \"06:00\"\n"
    )
    monkeypatch.setattr(advanced.prompts, "ask_confirm", lambda *a, **kw: True)
    monkeypatch.setattr(advanced.prompts, "ask_select", lambda *a, **kw: "Off")
    call_count = {"n": 0}

    def fake_text(message, default=None):
        call_count["n"] += 1
        if "category" in message.lower():
            return "5"
        return "100"

    monkeypatch.setattr(advanced.prompts, "ask_text", fake_text)

    write_calls = {}
    original_set = advanced.config_store.set_value

    def spy(path, key, value):
        write_calls[key] = value
        return original_set(path, key, value)

    monkeypatch.setattr(advanced.config_store, "set_value", spy)

    advanced.run_advanced_step(config_path=cfg)

    assert "triage.preflight_filter" in write_calls
    assert "briefing.limit_per_category" in write_calls
    assert "run.daily_limit" in write_calls


def test_advanced_step_rejects_negative_limit(tmp_path, monkeypatch):
    from argos.init_wizard import WizardAbort

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[triage]\npreflight_filter = true\n"
        "[briefing]\ntime = \"07:00\"\nweekdays = [\"Mon\"]\nlimit_per_category = 10\n"
        "[run]\ndaily_limit = 150\ntime = \"06:00\"\n"
    )
    monkeypatch.setattr(advanced.prompts, "ask_confirm", lambda *a, **kw: True)
    monkeypatch.setattr(advanced.prompts, "ask_select", lambda *a, **kw: "On")

    def fake_text(message, default=None):
        if "category" in message.lower():
            return "0"  # invalid: must be >= 1
        return "-1"  # invalid: must be >= 0

    monkeypatch.setattr(advanced.prompts, "ask_text", fake_text)

    with pytest.raises(WizardAbort):
        advanced.run_advanced_step(config_path=cfg)


def test_advanced_step_no_write_when_unchanged(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[triage]\npreflight_filter = true\n"
        "[briefing]\ntime = \"07:00\"\nweekdays = [\"Mon\"]\nlimit_per_category = 10\n"
        "[run]\ndaily_limit = 150\ntime = \"06:00\"\n"
    )
    monkeypatch.setattr(advanced.prompts, "ask_confirm", lambda *a, **kw: True)
    monkeypatch.setattr(advanced.prompts, "ask_select", lambda *a, **kw: "On")

    def fake_text(message, default=None):
        return default  # return current values unchanged

    monkeypatch.setattr(advanced.prompts, "ask_text", fake_text)

    write_calls = []
    monkeypatch.setattr(
        advanced.config_store, "set_value", lambda p, k, v: write_calls.append(k)
    )

    advanced.run_advanced_step(config_path=cfg)

    assert write_calls == []
