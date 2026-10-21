from __future__ import annotations

from argos.init_wizard.steps import schedule as schedule_step


def test_schedule_step_calls_reload_when_available(monkeypatch):
    seen = {}

    def fake_reload(user_config):
        seen["user_config"] = user_config

    monkeypatch.setattr(schedule_step, "reload_schedule", fake_reload)
    schedule_step.run_schedule_step({"slack": {}})
    assert seen["user_config"] == {"slack": {}}


def test_schedule_step_skips_and_warns_when_module_missing(monkeypatch, capsys):
    monkeypatch.setattr(schedule_step, "reload_schedule", None)
    result = schedule_step.run_schedule_step({})
    assert result is None
    captured = capsys.readouterr()
    assert "skipping" in captured.out


def test_schedule_step_converts_scheduler_error_to_wizard_step_error(monkeypatch):
    import pytest

    from argos.init_wizard import WizardStepError
    from argos.scheduler import SchedulerError

    def fake_reload(user_config):
        raise SchedulerError("correction.time 03:00 overlaps briefing.time")

    monkeypatch.setattr(schedule_step, "reload_schedule", fake_reload)
    with pytest.raises(WizardStepError, match="overlaps") as excinfo:
        schedule_step.run_schedule_step({})
    assert "correction.time" in (excinfo.value.hint or "")
