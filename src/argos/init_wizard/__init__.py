"""Interactive bootstrap wizard for Argos (``argos init``).

The package is organised so each of the six steps in the wizard
(precheck → infra → slack → interests → schedule → healthcheck) lives in
its own module under :mod:`argos.init_wizard.steps`, with all external
side effects (subprocess calls, HTTP probes, DB pings) funnelled through
:mod:`argos.init_wizard.runners` so tests can stub them cleanly.
"""

from __future__ import annotations


class WizardAbort(RuntimeError):
    """Precondition or validation failure that should exit non-zero (code 1).

    Raised when a real failure prevents the wizard from completing: missing
    required binaries, validation loop exhausted, non-interactive mode running
    out of valid defaults, etc.  Automation and CI can detect this as a failed
    run via the non-zero exit code.

    For explicit user-initiated cancellations (e.g. Ctrl-C at a prompt) use
    the :class:`WizardCancel` subclass, which exits with code 0.
    """


class WizardCancel(WizardAbort):
    """Explicit user-initiated cancel that should exit cleanly (code 0).

    Raised when the user consciously aborts the wizard (e.g. Ctrl-C / EOF at
    an interactive prompt).  Unlike the base :class:`WizardAbort`, this is not
    a failure — the user simply chose not to continue — so the process exits
    with code 0.
    """


class WizardStepError(RuntimeError):
    """A step failed in a way that should surface an actionable hint and exit non-zero."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


__all__ = ["WizardAbort", "WizardCancel", "WizardStepError"]
