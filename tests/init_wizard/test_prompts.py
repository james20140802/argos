from __future__ import annotations

import pytest

from argos.init_wizard import WizardAbort, prompts


@pytest.fixture(autouse=True)
def _force_noninteractive(monkeypatch):
    """Most prompt tests run in non-interactive mode unless flipped per-test."""
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")


def test_is_noninteractive_picks_up_env_var(monkeypatch):
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")
    assert prompts.is_noninteractive() is True

    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "0")
    assert prompts.is_noninteractive() is False


def test_ask_text_returns_default_when_noninteractive():
    assert prompts.ask_text("name", default="argos") == "argos"
    assert prompts.ask_text("name", default=None) == ""


def test_ask_password_returns_default_when_noninteractive():
    assert prompts.ask_password("pw", default="hunter2") == "hunter2"
    assert prompts.ask_password("pw") == ""


def test_ask_confirm_returns_default_when_noninteractive():
    assert prompts.ask_confirm("yes?", default=True) is True
    assert prompts.ask_confirm("yes?", default=False) is False


def test_ask_select_returns_default_when_noninteractive():
    assert prompts.ask_select("lang", choices=["Korean", "English"], default="English") == "English"
    # Default missing → first choice
    assert prompts.ask_select("lang", choices=["Korean", "English"]) == "Korean"


def test_ask_select_rejects_empty_choices():
    with pytest.raises(ValueError):
        prompts.ask_select("x", choices=[])


def test_ask_checkbox_returns_filtered_defaults():
    result = prompts.ask_checkbox(
        "days",
        choices=["Mon", "Tue", "Wed"],
        default=["Mon", "Wed"],
    )
    assert result == ["Mon", "Wed"]


def test_ask_checkbox_defaults_to_all_when_default_is_none():
    result = prompts.ask_checkbox("days", choices=["Mon", "Tue", "Wed"])
    assert result == ["Mon", "Tue", "Wed"]


def test_validation_loop_returns_first_valid_value():
    calls = iter(["bad", "still bad", "good"])

    def fake_prompt():
        return next(calls)

    def validator(value):
        return None if value == "good" else "nope"

    assert prompts.with_validation_loop(fake_prompt, validator, max_attempts=3) == "good"


def test_validation_loop_aborts_after_max_attempts():
    def fake_prompt():
        return "bad"

    def validator(value):
        return "nope"

    with pytest.raises(WizardAbort):
        prompts.with_validation_loop(fake_prompt, validator, max_attempts=3)


def test_validation_loop_requires_positive_attempts():
    with pytest.raises(ValueError):
        prompts.with_validation_loop(lambda: "", lambda v: None, max_attempts=0)
