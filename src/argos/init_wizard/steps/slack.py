"""Step 3: collect Slack credentials, verify them, and persist.

The flow:

1. Read the existing ``SLACK_BOT_TOKEN`` / ``SLACK_APP_TOKEN`` from ``.env``
   and the current ``slack.channel_id`` from ``config.toml`` so the wizard
   can be re-run idempotently.
2. Prompt for any value the user wants to update. Tokens are masked when
   echoed back as defaults via ``_mask_token_value``.
3. Validate the bot token by calling Slack ``auth.test``; on failure, loop
   (capped at 3 attempts via :func:`prompts.with_sensitive_validation_loop`).
4. Write the tokens back to ``.env`` only if they changed, and the
   ``slack.channel_id`` to ``config.toml`` only if it changed.
"""

from __future__ import annotations

from pathlib import Path

from argos import config_store
from argos.config_store import _mask_token_value
from argos.init_wizard import prompts, runners
from argos.init_wizard.env_file import atomic_write_env, harden_env_file_mode, load_env, merge_env


def _mask_for_default(value: str) -> str:
    """Return a placeholder a user can recognise but not exfiltrate."""
    if not value:
        return ""
    masked = _mask_token_value(value)
    if isinstance(masked, str) and masked.endswith("***"):
        return masked
    # Fall back to a generic mask — never echo arbitrary token text back.
    return "***"


def _persist_tokens(env_path: Path, bot_token: str, app_token: str) -> None:
    """Write tokens to ``.env`` only when at least one differs from disk."""
    existing = load_env(env_path)
    updates: dict[str, str] = {}
    if existing.get("SLACK_BOT_TOKEN", "") != bot_token:
        updates["SLACK_BOT_TOKEN"] = bot_token
    if existing.get("SLACK_APP_TOKEN", "") != app_token:
        updates["SLACK_APP_TOKEN"] = app_token
    if not updates:
        harden_env_file_mode(env_path)
        return
    merged = merge_env(existing, updates)
    atomic_write_env(env_path, merged)


def _persist_channel_id(config_path: Path, current_channel_id: str, new_channel_id: str) -> None:
    """Write ``slack.channel_id`` to ``config.toml`` only when it changed."""
    if current_channel_id == new_channel_id:
        return
    config_store.set_value(config_path, "slack.channel_id", new_channel_id)


def run_slack_step(
    repo_root: Path,
    env_path: Path | None = None,
    config_path: Path | None = None,
) -> None:
    """Prompt for + validate + persist Slack credentials and channel ID."""
    from argos.config import UserConfig  # local import keeps test isolation cheap

    env_file = env_path if env_path is not None else (repo_root / ".env")
    cfg_file = config_path if config_path is not None else config_store.default_config_path()

    existing_env = load_env(env_file)
    current_bot = existing_env.get("SLACK_BOT_TOKEN", "")
    current_app = existing_env.get("SLACK_APP_TOKEN", "")
    current_cfg = UserConfig.load(path=cfg_file)
    current_channel = current_cfg.slack.channel_id

    bot_default_hint = _mask_for_default(current_bot)
    app_default_hint = _mask_for_default(current_app)

    def _prompt_bot() -> str:
        # The raw current_bot token must NOT flow into ask_password's
        # ``default=`` — CodeQL (rightly) traces that value back through the
        # validation loop into the printed error sink. Prompt with no default
        # and re-apply the existing value locally when the user submits empty.
        message = f"SLACK_BOT_TOKEN [{bot_default_hint or 'xoxb-…'}]"
        answer = prompts.ask_password(message)
        return answer or current_bot

    def _validate_bot(value: str) -> str | None:
        if not value:
            return "bot token is required"
        if not value.startswith("xoxb-"):
            return "bot tokens must start with 'xoxb-'"
        try:
            runners.slack_auth_test(value)
        except Exception:  # noqa: BLE001 — message intentionally discarded
            # The Slack SDK may echo the offending token back in its
            # exception message; returning ``str(exc)`` (even with
            # ``.replace()`` scrubbing) lets the raw secret reach a print
            # sink. Return a fixed message instead — ``with_sensitive_validation_loop``
            # discards the validator's string return anyway.
            return "slack auth.test rejected the token"
        return None

    bot_token = prompts.with_sensitive_validation_loop(
        _prompt_bot, _validate_bot, max_attempts=3
    )

    def _prompt_app() -> str:
        # See _prompt_bot — raw current_app must not flow into ask_password.
        message = f"SLACK_APP_TOKEN [{app_default_hint or 'xapp-…'}]"
        answer = prompts.ask_password(message)
        return answer or current_app

    def _validate_app(value: str) -> str | None:
        if not value:
            return "app token is required (Socket Mode connection needs it)"
        if not value.startswith("xapp-"):
            return "app tokens must start with 'xapp-'"
        try:
            runners.slack_app_connections_open(value)
        except Exception:  # noqa: BLE001 — message intentionally discarded
            # The Slack SDK or runner may echo the offending token back in its
            # exception message; returning a fixed string prevents any token
            # from reaching a print sink via the validation loop.
            return "slack apps.connections.open rejected the app token"
        return None

    app_token = prompts.with_sensitive_validation_loop(
        _prompt_app, _validate_app, max_attempts=3
    )

    channel_default = current_channel or "C01234567"
    channel_id = prompts.ask_text(
        f"Slack channel ID [{channel_default}]",
        default=current_channel,
    )

    _persist_tokens(env_file, bot_token, app_token)
    _persist_channel_id(cfg_file, current_channel, channel_id)


__all__ = ["run_slack_step"]
