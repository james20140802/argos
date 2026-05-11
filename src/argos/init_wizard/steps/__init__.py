"""Wizard step modules.

Each submodule exposes one public entry point named ``run_<step>_step`` that
takes whatever state it needs (paths, UserConfig, etc.) and either returns
``None`` on success or raises :class:`argos.init_wizard.WizardAbort` /
:class:`argos.init_wizard.WizardStepError`. The wizard orchestrator
(:mod:`argos.init_wizard.wizard`) wires them together in order.
"""
