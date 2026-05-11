"""Top-level orchestrator for ``argos init``.

``run_full()`` walks the six steps in order; ``run_reconfigure(section)``
dispatches to a single step (plus a trailing healthcheck) so users can
re-run, say, only the Slack flow without re-prompting for Postgres.

Both entry points are synchronous and return an integer process exit code
so :func:`argos.cli.main` can ``sys.exit(...)`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from argos import config_store
from argos.init_wizard import WizardAbort, WizardStepError
from argos.init_wizard.steps.healthcheck import run_healthcheck_step
from argos.init_wizard.steps.infra import run_infra_step
from argos.init_wizard.steps.interests import run_interests_step
from argos.init_wizard.steps.precheck import run_precheck_step
from argos.init_wizard.steps.schedule import run_schedule_step
from argos.init_wizard.steps.slack import run_slack_step

logger = logging.getLogger(__name__)

# Mapping of --reconfigure section name → (header, runner factory).
# Each runner factory takes (repo_root, env_path, config_path) and returns a
# zero-arg callable so we can pass extra state in without leaking it into the
# public signature of each step.
RECONFIGURE_SECTIONS: tuple[str, ...] = ("infra", "slack", "interests", "schedule")


def _repo_root() -> Path:
    """Best-effort detection of the Argos repo root.

    We walk up from this file looking for ``docker-compose.yml``. In a normal
    install this resolves to the worktree; in tests this is overridden via the
    ``repo_root`` kwarg on each entry point.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "docker-compose.yml").exists():
            return parent
    return Path.cwd()


def _print_header(step_no: int, total: int, title: str) -> None:
    print(f"\n[{step_no}/{total}] {title}")
    print("─" * 40)


def _user_config():
    from argos.config import UserConfig

    return UserConfig.load(path=config_store.default_config_path())


def _handle(exc: BaseException) -> int:
    """Translate a :class:`WizardAbort` / :class:`WizardStepError` into an exit code."""
    if isinstance(exc, WizardAbort):
        print(f"\n{exc}")
        return 0
    if isinstance(exc, WizardStepError):
        print(f"\nerror: {exc}")
        if exc.hint:
            print(f"hint: {exc.hint}")
        return 1
    raise exc  # let the caller see anything unexpected


def run_full(
    repo_root: Path | None = None,
    env_path: Path | None = None,
    config_path: Path | None = None,
) -> int:
    """Walk every step in order. Returns the process exit code."""
    root = repo_root or _repo_root()
    cfg_path = config_path or config_store.default_config_path()

    total = 6
    try:
        _print_header(1, total, "Precheck — verifying required binaries")
        run_precheck_step()

        _print_header(2, total, "Infra — Postgres + Alembic + Ollama")
        run_infra_step(root, env_path=env_path)

        _print_header(3, total, "Slack — bot token + channel")
        run_slack_step(root, env_path=env_path, config_path=cfg_path)

        _print_header(4, total, "Interests — language, topics, schedule")
        run_interests_step(config_path=cfg_path)

        _print_header(5, total, "Schedule — launchd plists")
        run_schedule_step(_user_config())

        _print_header(6, total, "Healthcheck")
        failures = run_healthcheck_step(root, env_path=env_path)
        if failures:
            print(f"\n{failures} healthcheck probe(s) failed — see above")
            return 1
        print("\n✓ argos init complete")
        return 0
    except (WizardAbort, WizardStepError) as exc:
        return _handle(exc)


def run_reconfigure(
    section: str,
    repo_root: Path | None = None,
    env_path: Path | None = None,
    config_path: Path | None = None,
) -> int:
    """Run a single section plus a trailing healthcheck."""
    if section not in RECONFIGURE_SECTIONS:
        raise ValueError(
            f"unknown reconfigure section {section!r} — choose one of {list(RECONFIGURE_SECTIONS)}"
        )
    root = repo_root or _repo_root()
    cfg_path = config_path or config_store.default_config_path()

    try:
        _print_header(1, 2, f"Reconfigure — {section}")
        if section == "infra":
            run_infra_step(root, env_path=env_path)
        elif section == "slack":
            run_slack_step(root, env_path=env_path, config_path=cfg_path)
        elif section == "interests":
            run_interests_step(config_path=cfg_path)
        elif section == "schedule":
            run_schedule_step(_user_config())

        _print_header(2, 2, "Healthcheck")
        failures = run_healthcheck_step(root, env_path=env_path)
        if failures:
            print(f"\n{failures} healthcheck probe(s) failed — see above")
            return 1
        print(f"\n✓ argos init --reconfigure {section} complete")
        return 0
    except (WizardAbort, WizardStepError) as exc:
        return _handle(exc)


__all__ = ["RECONFIGURE_SECTIONS", "run_full", "run_reconfigure"]
