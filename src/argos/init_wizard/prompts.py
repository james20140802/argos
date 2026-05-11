"""Thin wrappers around :mod:`questionary` that the wizard steps call.

Centralising the prompt API has two benefits:

* Unit tests monkeypatch one module instead of stubbing ``questionary`` directly.
* The ``ARGOS_INIT_NONINTERACTIVE=1`` env var (used by CI / piped stdin /
  the ``--non-interactive`` CLI flag) is honoured in exactly one place — the
  wrappers fall back to the supplied default silently.

The wrappers intentionally accept the same ``default=`` semantics across all
five primitives so step modules can write::

    name = prompts.ask_text("Project name", default=existing_value)

and not worry about whether the user pressed Enter, piped data in, or set
the env var.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import questionary

from argos.init_wizard import WizardAbort

_NONINTERACTIVE_ENV = "ARGOS_INIT_NONINTERACTIVE"


def is_noninteractive() -> bool:
    """Return True when prompts should silently fall back to defaults."""
    return os.environ.get(_NONINTERACTIVE_ENV, "").strip() in {"1", "true", "yes", "on"}


def _coerce_text_default(default: Any) -> str:
    if default is None:
        return ""
    return str(default)


def ask_text(message: str, *, default: str | None = None) -> str:
    """Free-form text prompt. Returns ``default`` (or "") in non-interactive mode."""
    fallback = _coerce_text_default(default)
    if is_noninteractive():
        return fallback
    answer = questionary.text(message, default=fallback).ask()
    if answer is None:  # User hit Ctrl-C
        raise WizardAbort("user cancelled prompt")
    return answer


def ask_password(message: str, *, default: str | None = None) -> str:
    """Masked password prompt. Returns ``default`` (or "") in non-interactive mode.

    questionary's password widget doesn't accept ``default=``; we surface a
    ``(unchanged)`` hint in the message text instead, and substitute the
    default when the user submits an empty string.
    """
    fallback = _coerce_text_default(default)
    if is_noninteractive():
        return fallback
    answer = questionary.password(message).ask()
    if answer is None:
        raise WizardAbort("user cancelled prompt")
    return answer or fallback


def ask_confirm(message: str, *, default: bool = True) -> bool:
    """Yes/no prompt. Returns ``default`` in non-interactive mode."""
    if is_noninteractive():
        return default
    answer = questionary.confirm(message, default=default).ask()
    if answer is None:
        raise WizardAbort("user cancelled prompt")
    return bool(answer)


def ask_select(
    message: str,
    *,
    choices: Sequence[str],
    default: str | None = None,
) -> str:
    """Single-choice prompt. Returns ``default`` (or the first choice) when non-interactive."""
    if not choices:
        raise ValueError("ask_select requires a non-empty choices list")
    if is_noninteractive():
        if default is not None and default in choices:
            return default
        return choices[0]
    answer = questionary.select(message, choices=list(choices), default=default).ask()
    if answer is None:
        raise WizardAbort("user cancelled prompt")
    return answer


def ask_checkbox(
    message: str,
    *,
    choices: Sequence[str],
    default: Sequence[str] | None = None,
) -> list[str]:
    """Multi-select prompt. Returns ``default`` (or all choices) when non-interactive."""
    defaults = list(default) if default else list(choices)
    if is_noninteractive():
        return [c for c in choices if c in defaults]
    # questionary's checkbox uses per-choice `checked` flags rather than `default=`.
    q_choices = [questionary.Choice(c, checked=(c in defaults)) for c in choices]
    answer = questionary.checkbox(message, choices=q_choices).ask()
    if answer is None:
        raise WizardAbort("user cancelled prompt")
    return list(answer)


def with_validation_loop(
    prompt_fn: Callable[[], str],
    validator: Callable[[str], str | None],
    *,
    max_attempts: int = 3,
) -> str:
    """Re-invoke ``prompt_fn`` until ``validator`` returns ``None`` (= valid).

    ``validator`` returns ``None`` on success or an error message string. The
    loop is capped at ``max_attempts`` (default 3) — exceeding the cap raises
    :class:`argos.init_wizard.WizardAbort` so callers exit cleanly.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        value = prompt_fn()
        error = validator(value)
        if error is None:
            return value
        last_error = error
        print(f"  ✗ {error} (attempt {attempt}/{max_attempts})")
    raise WizardAbort(f"validation failed after {max_attempts} attempts: {last_error}")


__all__ = [
    "ask_checkbox",
    "ask_confirm",
    "ask_password",
    "ask_select",
    "ask_text",
    "is_noninteractive",
    "with_validation_loop",
]
