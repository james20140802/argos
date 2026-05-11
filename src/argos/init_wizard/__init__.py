"""Interactive bootstrap wizard for Argos (``argos init``).

The package is organised so each of the six steps in the wizard
(precheck → infra → slack → interests → schedule → healthcheck) lives in
its own module under :mod:`argos.init_wizard.steps`, with all external
side effects (subprocess calls, HTTP probes, DB pings) funnelled through
:mod:`argos.init_wizard.runners` so tests can stub them cleanly.
"""

from __future__ import annotations


class WizardAbort(RuntimeError):
    """User-initiated abort or precondition failure that should exit cleanly (code 0)."""


class WizardStepError(RuntimeError):
    """A step failed in a way that should surface an actionable hint and exit non-zero."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


__all__ = ["WizardAbort", "WizardStepError"]
