from __future__ import annotations

import pytest

from argos.init_wizard import WizardStepError
from argos.init_wizard.steps import schedule as schedule_step


def test_schedule_step_calls_reload_when_available(monkeypatch):
    seen = {}

    def fake_reload(user_config):
        seen["user_config"] = user_config

    monkeypatch.setattr(schedule_step, "reload_schedule", fake_reload)
    schedule_step.run_schedule_step({"slack": {}})
    assert seen["user_config"] == {"slack": {}}


def test_schedule_step_raises_when_module_missing(monkeypatch):
    monkeypatch.setattr(schedule_step, "reload_schedule", None)
    with pytest.raises(WizardStepError) as excinfo:
        schedule_step.run_schedule_step({})
    assert "scheduler module not available" in str(excinfo.value)
    assert excinfo.value.hint and "ARG-51" in excinfo.value.hint
