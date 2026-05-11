"""Step 5: render + install + bootstrap the launchd plists via ARG-51.

The scheduler module lives in ARG-51 (a parallel PR) — until it lands,
this step degrades gracefully: it prints a warning and returns early so the
rest of the wizard (infra, slack, interests, healthcheck) can still complete.
Once ARG-51 merges, the ``try`` import succeeds and the step delegates
everything to ``argos.scheduler.reload_schedule``.

Tests monkeypatch ``argos.init_wizard.steps.schedule.reload_schedule`` (the
name bound *after* the try/except) so they never depend on ARG-51 being
importable.
"""

from __future__ import annotations

try:
    from argos.scheduler import reload_schedule  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised once ARG-51 merges
    reload_schedule = None  # type: ignore[assignment]


def run_schedule_step(user_config) -> None:  # type: ignore[no-untyped-def]
    """Install the launchd plists for ``argos run`` and ``argos brief``."""
    if reload_schedule is None:
        print(
            "  ⚠ scheduler module not available — skipping schedule setup.\n"
            "    (Install ARG-51’s argos.scheduler module to enable launchd schedules.)"
        )
        return
    reload_schedule(user_config)


__all__ = ["run_schedule_step"]
