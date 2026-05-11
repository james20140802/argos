"""Step 4: collect briefing language, topics, exclusions, and schedule.

Despite the module name, the step writes to **three** config sections so the
wizard can present a single coherent "what do you want briefed?" screen:

* ``slack.summary_language`` — language the brain summarises items in.
* ``interests.topics`` / ``interests.exclusions`` — comma-separated lists.
* ``briefing.time`` (HH:MM) and ``briefing.weekdays`` (subset of Mon-Sun).

Each value is written via :func:`argos.config_store.set_value` only when it
differs from the current on-disk value so reruns don't churn the TOML file.
"""

from __future__ import annotations

import re
from pathlib import Path

from argos import config_store
from argos.init_wizard import prompts

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

_LANGUAGE_CHOICES: tuple[str, ...] = (
    "Korean",
    "English",
    "Japanese",
    "Chinese",
    "Custom…",
)

_WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _split_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _join_csv(values: list[str]) -> str:
    return ", ".join(values)


def _maybe_set(
    config_path: Path,
    dotted_key: str,
    current: object,
    new_value: object,
) -> None:
    """Write ``new_value`` only when it differs from ``current``."""
    if current == new_value:
        return
    if isinstance(new_value, list):
        config_store.set_value(config_path, dotted_key, ",".join(str(v) for v in new_value))
    else:
        config_store.set_value(config_path, dotted_key, str(new_value))


def _prompt_language(current: str) -> str:
    default_choice = current if current in _LANGUAGE_CHOICES else "Custom…"
    choice = prompts.ask_select(
        f"Summary language [{current or 'Korean'}]",
        choices=list(_LANGUAGE_CHOICES),
        default=default_choice,
    )
    if choice != "Custom…":
        return choice
    custom = prompts.ask_text(
        "Custom language name",
        default=current if current not in _LANGUAGE_CHOICES else "",
    )
    return custom or current or "Korean"


def _prompt_time(current: str) -> str:
    def _prompt() -> str:
        return prompts.ask_text(
            f"Briefing time HH:MM [{current or '07:00'}]",
            default=current or "07:00",
        )

    def _validate(value: str) -> str | None:
        if not _HHMM_RE.match(value):
            return f"{value!r} is not a valid HH:MM (00:00-23:59) string"
        return None

    return prompts.with_validation_loop(_prompt, _validate, max_attempts=3)


def _prompt_weekdays(current: list[str]) -> list[str]:
    defaults = [d for d in _WEEKDAYS if d in current] or list(_WEEKDAYS)
    return prompts.ask_checkbox(
        "Briefing weekdays",
        choices=list(_WEEKDAYS),
        default=defaults,
    )


def run_interests_step(config_path: Path | None = None) -> None:
    """Drive the briefing-preferences sub-flow."""
    from argos.config import UserConfig

    cfg_file = config_path if config_path is not None else config_store.default_config_path()
    cfg = UserConfig.load(path=cfg_file)

    new_language = _prompt_language(cfg.slack.summary_language)

    raw_topics = prompts.ask_text(
        f"Topics (comma-separated) [{_join_csv(cfg.interests.topics) or '(none)'}]",
        default=_join_csv(cfg.interests.topics),
    )
    new_topics = _split_csv(raw_topics)

    raw_excl = prompts.ask_text(
        f"Exclusions (comma-separated) [{_join_csv(cfg.interests.exclusions) or '(none)'}]",
        default=_join_csv(cfg.interests.exclusions),
    )
    new_exclusions = _split_csv(raw_excl)

    new_time = _prompt_time(cfg.briefing.time)
    new_weekdays = _prompt_weekdays(cfg.briefing.weekdays)

    _maybe_set(cfg_file, "slack.summary_language", cfg.slack.summary_language, new_language)
    _maybe_set(cfg_file, "interests.topics", cfg.interests.topics, new_topics)
    _maybe_set(cfg_file, "interests.exclusions", cfg.interests.exclusions, new_exclusions)
    _maybe_set(cfg_file, "briefing.time", cfg.briefing.time, new_time)
    _maybe_set(cfg_file, "briefing.weekdays", cfg.briefing.weekdays, new_weekdays)


__all__ = ["run_interests_step"]
