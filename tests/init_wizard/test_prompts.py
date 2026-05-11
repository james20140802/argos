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


def test_mask_secret_preserves_slack_prefixes():
    assert prompts.mask_secret("xoxb-1234567890-secretvalue") == "xoxb-***"
    assert prompts.mask_secret("xapp-1-A1B2C3-tokenbody") == "xapp-***"


def test_mask_secret_generic_fallback():
    assert prompts.mask_secret("hunter2") == "***"
    assert prompts.mask_secret("") == "***"


def test_validation_loop_sensitive_scrubs_value_from_printed_error(capsys):
    secret = "xoxb-leak-me-1234567890"

    def fake_prompt():
        return secret

    def validator(value):
        # Mimics an SDK echoing the offending token back in its error message.
        return f"invalid_auth for token {value}"

    with pytest.raises(WizardAbort) as excinfo:
        prompts.with_sensitive_validation_loop(
            fake_prompt, validator, max_attempts=2
        )

    captured = capsys.readouterr()
    # Raw secret must never reach stdout/stderr…
    assert secret not in captured.out
    assert secret not in captured.err
    # …nor the abort message that callers may log…
    assert secret not in str(excinfo.value)
    # …nor any field on the raised exception (args, repr).
    assert all(secret not in str(a) for a in excinfo.value.args)
    assert secret not in repr(excinfo.value)
    # A fixed redacted notice should be visible so the user still has context.
    assert "redacted" in captured.out


def test_validation_loop_sensitive_never_leaks_after_three_failures(
    capsys, caplog, monkeypatch
):
    """Regression for CodeQL alert #2 — covers the password-input sink chain.

    Simulates a questionary password prompt returning a sentinel three times,
    then asserts the sentinel never reaches stdout, stderr, log records, or
    the raised ``WizardAbort`` exception's ``args``/``repr``.
    """
    import logging

    sentinel = "SECRET_SENTINEL_xyz123"
    call_count = {"n": 0}

    def fake_questionary_prompt():
        call_count["n"] += 1
        return sentinel

    # Validator deliberately interpolates the value (mimics a real SDK leak).
    def validator(value):
        return f"backend rejected token {value!r}"

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(WizardAbort) as excinfo:
            prompts.with_sensitive_validation_loop(
                fake_questionary_prompt,
                validator,
                max_attempts=3,
            )

    assert call_count["n"] == 3  # loop ran the full 3 attempts

    captured = capsys.readouterr()
    # Sinks that CodeQL traced from the password input must never see the
    # raw value.
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    for record in caplog.records:
        assert sentinel not in record.getMessage()
        assert sentinel not in str(record.args or "")
    assert sentinel not in str(excinfo.value)
    assert all(sentinel not in str(a) for a in excinfo.value.args)
    assert sentinel not in repr(excinfo.value)


def test_validation_loop_sensitive_has_no_error_binding_in_source():
    """Structural guard for CodeQL — the sensitive loop must not bind the
    validator's return value to a string variable (e.g. ``error``), otherwise
    taint analysis can construct a flow from the raw secret into a sink.

    This test inspects the source of :func:`_validation_loop_sensitive` and
    asserts that no ``error`` binding exists. Pair it with the runtime leak
    test above for defence-in-depth.
    """
    import inspect

    source = inspect.getsource(prompts._validation_loop_sensitive)
    # The token ``error`` must not appear as an identifier anywhere in the
    # sensitive loop's body — neither as a left-hand side, an f-string slot,
    # nor an exception-message component.
    assert "error" not in source, (
        "sensitive validation loop must not bind validator's string return "
        "to an `error` variable — see CodeQL clear-text-logging guidance"
    )


def test_validation_loop_non_sensitive_still_prints_error_verbatim(capsys):
    def fake_prompt():
        return "not-a-secret"

    def validator(value):
        return "value must be foo"

    with pytest.raises(WizardAbort):
        prompts.with_validation_loop(fake_prompt, validator, max_attempts=1)

    captured = capsys.readouterr()
    assert "value must be foo" in captured.out
