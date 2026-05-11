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


def mask_secret(value: str) -> str:
    """Return a redacted placeholder for a secret value.

    Recognised Slack token prefixes (``xoxb-`` / ``xapp-``) are preserved so a
    user can still tell what kind of credential was rejected, but the
    body is replaced with ``***``. Anything else collapses to ``***``.
    """
    if not isinstance(value, str) or not value:
        return "***"
    for prefix in ("xoxb-", "xapp-"):
        if value.startswith(prefix):
            return f"{prefix}***"
    return "***"


_SENSITIVE_GENERIC_ERROR = "validation failed (details redacted to avoid logging secrets)"


def _validation_loop_plain(
    prompt_fn: Callable[[], str],
    validator: Callable[[str], str | None],
    *,
    max_attempts: int,
) -> str:
    """Plain validation loop — validator's error string flows into the printed
    line and the abort message. Only safe for non-sensitive values."""
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        value = prompt_fn()
        error = validator(value)
        if error is None:
            return value
        last_error = error
        print(f"  ✗ {error} (attempt {attempt}/{max_attempts})")
    raise WizardAbort(
        f"validation failed after {max_attempts} attempts (last error: {last_error})"
    )


def _validation_loop_sensitive(
    prompt_fn: Callable[[], str],
    validator: Callable[[str], str | None],
    *,
    max_attempts: int,
) -> str:
    """Sensitive validation loop — the validator's return value is consumed as
    a boolean *only*; its string contents (which may embed the raw secret) are
    never bound to a variable and therefore cannot flow into any sink.

    This is the structural defence against CodeQL's "clear-text logging of
    sensitive information" taint analysis: because there is no assignment of
    ``validator(value)`` to a string-typed binding inside this function, the
    taint analyser cannot construct a flow path from the password source to
    a print/log/exception-message sink.
    """
    for attempt in range(1, max_attempts + 1):
        value = prompt_fn()
        ok = validator(value) is None  # consume as bool only — never bind the string
        if ok:
            return value
        print(f"  ✗ {_SENSITIVE_GENERIC_ERROR} (attempt {attempt}/{max_attempts})")
    raise WizardAbort(
        f"validation failed after {max_attempts} attempts ({_SENSITIVE_GENERIC_ERROR})"
    )


def with_validation_loop(
    prompt_fn: Callable[[], str],
    validator: Callable[[str], str | None],
    *,
    max_attempts: int = 3,
) -> str:
    """Re-invoke ``prompt_fn`` until ``validator`` returns ``None`` (= valid).

    For **non-sensitive** prompts only. ``validator`` returns ``None`` on
    success or an error message string; the error string is printed verbatim
    and embedded in the final :class:`argos.init_wizard.WizardAbort` message.
    The loop is capped at ``max_attempts`` (default 3) — exceeding the cap
    raises ``WizardAbort`` so callers exit cleanly.

    For password / token prompts use :func:`with_sensitive_validation_loop`
    instead. That function is a separate, statically distinct entry point
    so taint analysers (CodeQL) can prove the plain-text printing loop is
    unreachable from sensitive call sites.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    return _validation_loop_plain(
        prompt_fn, validator, max_attempts=max_attempts
    )


def with_sensitive_validation_loop(
    prompt_fn: Callable[[], str],
    validator: Callable[[str], str | None],
    *,
    max_attempts: int = 3,
) -> str:
    """Validation loop for password / token prompts.

    Behaves like :func:`with_validation_loop` but routes to
    :func:`_validation_loop_sensitive`, which consumes the validator's return
    value as a boolean only — its string contents (which may embed the raw
    secret) are never bound to a local variable, so static taint analysers
    (CodeQL) cannot construct a flow path from the password source to a
    print/log/exception-message sink.

    Validators for sensitive flows should still return a non-``None`` truthy
    sentinel on failure (any non-empty string works) to signal "retry"; the
    string is never read.

    This is intentionally a *separate* public function rather than a flag on
    :func:`with_validation_loop`: a runtime ``if sensitive`` dispatcher leaves
    both helpers reachable from every call site as far as CodeQL is concerned,
    so the plain (verbatim-printing) helper still shows up as a sink for
    password sources. With two distinct entry points, each statically points
    at exactly one helper.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    return _validation_loop_sensitive(
        prompt_fn, validator, max_attempts=max_attempts
    )


__all__ = [
    "ask_checkbox",
    "ask_confirm",
    "ask_password",
    "ask_select",
    "ask_text",
    "is_noninteractive",
    "mask_secret",
    "with_sensitive_validation_loop",
    "with_validation_loop",
]
