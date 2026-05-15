from __future__ import annotations

from pathlib import Path

from argos import config_store
from argos.init_wizard import prompts


def _ask_preflight_filter(current: bool) -> bool:
    default = "On" if current else "Off"
    choice = prompts.ask_select(
        f"Heuristic pre-filter — removes job ads/marketing before LLM [{default}]",
        choices=["On", "Off"],
        default=default,
    )
    return choice == "On"


def _ask_limit_per_category(current: int) -> int:
    def _prompt() -> str:
        return prompts.ask_text(
            f"Briefing items per category (≥ 1) [{current}]",
            default=str(current),
        )

    def _validate(value: str) -> str | None:
        try:
            n = int(value)
        except ValueError:
            return f"{value!r} is not a valid integer"
        if n < 1:
            return "Must be at least 1"
        return None

    return int(prompts.with_validation_loop(_prompt, _validate, max_attempts=3))


def _ask_daily_limit(current: int) -> int:
    def _prompt() -> str:
        return prompts.ask_text(
            f"Daily crawl limit — 0 = unlimited [{current}]",
            default=str(current),
        )

    def _validate(value: str) -> str | None:
        try:
            n = int(value)
        except ValueError:
            return f"{value!r} is not a valid integer"
        if n < 0:
            return "Must be 0 or greater"
        return None

    return int(prompts.with_validation_loop(_prompt, _validate, max_attempts=3))


def run_advanced_step(config_path: Path | None = None) -> None:
    from argos.config import UserConfig

    cfg_file = config_path if config_path is not None else config_store.default_config_path()

    if not prompts.ask_confirm("Configure advanced settings?", default=False):
        return

    cfg = UserConfig.load(path=cfg_file)

    new_preflight = _ask_preflight_filter(cfg.triage.preflight_filter)
    new_limit = _ask_limit_per_category(cfg.briefing.limit_per_category)
    new_daily = _ask_daily_limit(cfg.run.daily_limit)

    if new_preflight != cfg.triage.preflight_filter:
        config_store.set_value(cfg_file, "triage.preflight_filter", str(new_preflight).lower())
    if new_limit != cfg.briefing.limit_per_category:
        config_store.set_value(cfg_file, "briefing.limit_per_category", str(new_limit))
    if new_daily != cfg.run.daily_limit:
        config_store.set_value(cfg_file, "run.daily_limit", str(new_daily))


__all__ = ["run_advanced_step"]
