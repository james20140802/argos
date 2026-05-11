from __future__ import annotations

import pytest

from argos.init_wizard import WizardAbort
from argos.init_wizard.steps import interests


@pytest.fixture(autouse=True)
def _force_noninteractive(monkeypatch):
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")


def test_interests_step_writes_changed_values(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[slack]\nsummary_language = "Korean"\n'
        '[briefing]\ntime = "07:00"\nweekdays = ["Mon", "Tue"]\nlimit_per_category = 10\n'
        '[interests]\ntopics = []\nexclusions = []\n'
    )

    # Override the prompts so we're not at the mercy of which default the
    # non-interactive selectors fall back to.
    monkeypatch.setattr(interests.prompts, "ask_select", lambda *a, **kw: "English")
    monkeypatch.setattr(
        interests.prompts,
        "ask_text",
        lambda message, default=None: {
            "Topics": "rust, python",
            "Exclusions": "blockchain",
            "Time": "09:00",
        }[next(k for k in ("Topics", "Exclusions", "Time") if k.lower() in message.lower())],
    )
    monkeypatch.setattr(
        interests.prompts,
        "ask_checkbox",
        lambda *a, **kw: ["Mon", "Wed", "Fri"],
    )

    write_calls = []
    original_set = interests.config_store.set_value

    def spy(path, key, value):
        write_calls.append((key, value))
        return original_set(path, key, value)

    monkeypatch.setattr(interests.config_store, "set_value", spy)
    interests.run_interests_step(config_path=cfg)

    keys = {k for k, _ in write_calls}
    assert "slack.summary_language" in keys
    assert "interests.topics" in keys
    assert "briefing.time" in keys
    assert "briefing.weekdays" in keys


def test_interests_step_rejects_bad_hhmm(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[slack]\nsummary_language = "Korean"\n'
        '[briefing]\ntime = "07:00"\nweekdays = ["Mon"]\nlimit_per_category = 10\n'
        '[interests]\ntopics = []\nexclusions = []\n'
    )

    monkeypatch.setattr(interests.prompts, "ask_select", lambda *a, **kw: "Korean")
    monkeypatch.setattr(interests.prompts, "ask_checkbox", lambda *a, **kw: ["Mon"])
    # First three calls feed topics, exclusions, then bad HH:MM repeatedly.
    call_log = []

    def fake_text(message, default=None):
        call_log.append(message)
        lower = message.lower()
        if "topic" in lower:
            return ""
        if "exclusion" in lower:
            return ""
        return "25:99"  # invalid for all subsequent calls

    monkeypatch.setattr(interests.prompts, "ask_text", fake_text)

    with pytest.raises(WizardAbort):
        interests.run_interests_step(config_path=cfg)


def test_interests_step_topics_csv_round_trip(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[slack]\nsummary_language = "Korean"\n'
        '[briefing]\ntime = "07:00"\nweekdays = ["Mon"]\nlimit_per_category = 10\n'
        '[interests]\ntopics = []\nexclusions = []\n'
    )

    monkeypatch.setattr(interests.prompts, "ask_select", lambda *a, **kw: "Korean")
    monkeypatch.setattr(interests.prompts, "ask_checkbox", lambda *a, **kw: ["Mon"])
    monkeypatch.setattr(
        interests.prompts,
        "ask_text",
        lambda message, default=None: (
            "rust, python, llm" if "topic" in message.lower() else
            "" if "exclusion" in message.lower() else
            "07:00"
        ),
    )

    interests.run_interests_step(config_path=cfg)

    from argos.config import UserConfig
    cfg_loaded = UserConfig.load(path=cfg)
    assert cfg_loaded.interests.topics == ["rust", "python", "llm"]
