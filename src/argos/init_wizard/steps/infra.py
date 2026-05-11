"""Step 2: bring up Postgres + run migrations + ensure required Ollama models.

The step is broken into three sub-phases:

1. Prompt for the ``POSTGRES_*`` values (defaulting to whatever is already in
   ``.env``), write the merged ``.env`` atomically only if the user changed
   something. Secrets are masked when re-displayed as defaults.
2. ``docker compose up -d`` + poll ``pg_isready`` for up to 30 s, then run
   ``alembic upgrade head``.
3. Diff ``ollama list`` against the three models Argos requires
   (``qwen3:8b``, ``qwen3:32b``, ``nomic-embed-text``) and ``ollama pull``
   only the missing ones.
"""

from __future__ import annotations

from pathlib import Path

from argos.config_store import _mask_token_value  # reuse the existing masker
from argos.init_wizard import WizardStepError, prompts, runners
from argos.init_wizard.env_file import atomic_write_env, harden_env_file_mode, load_env, merge_env

REQUIRED_OLLAMA_MODELS: tuple[str, ...] = ("qwen3:8b", "qwen3:32b", "nomic-embed-text")

_PG_KEYS: tuple[str, ...] = (
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
)

_PG_DEFAULTS: dict[str, str] = {
    "POSTGRES_USER": "argos",
    "POSTGRES_PASSWORD": "argos_dev_password",
    "POSTGRES_DB": "argos",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
}


def _mask_for_display(key: str, value: str) -> str:
    """Return a value suitable for showing back as a prompt default.

    Password fields are masked entirely (we never echo a secret in plaintext);
    other fields fall back to ``_mask_token_value`` which detects ``xoxb-`` /
    ``xapp-`` prefixes — relevant if a user mis-pastes a Slack token into a PG
    field by accident.
    """
    if "PASSWORD" in key or "SECRET" in key or "TOKEN" in key:
        return "***" if value else ""
    return str(_mask_token_value(value))


def _validate_port(raw: str) -> str | None:
    """Validator for ``POSTGRES_PORT``: must parse as int and fall in 1..65535.

    Returns ``None`` on success or a short error message describing why the
    value is rejected. The string is wired into
    :func:`prompts.with_validation_loop`, which prints the error verbatim and
    re-prompts up to ``max_attempts`` times.
    """
    stripped = raw.strip()
    if not stripped:
        return "port is required"
    try:
        port = int(stripped)
    except ValueError:
        return f"not a number: {stripped!r}"
    if not (1 <= port <= 65535):
        return f"out of range (1-65535): {port}"
    return None


def _prompt_pg_values(existing: dict[str, str]) -> dict[str, str]:
    """Walk through the PG_* keys and collect (possibly-new) values from the user."""
    updates: dict[str, str] = {}
    for key in _PG_KEYS:
        current = existing.get(key) or _PG_DEFAULTS[key]
        masked = _mask_for_display(key, current)
        message = f"{key} [{masked}]"
        if "PASSWORD" in key:
            value = prompts.ask_password(message, default=current)
        elif key == "POSTGRES_PORT" and not prompts.is_noninteractive():
            # Interactive: re-prompt on invalid input so a single typo doesn't
            # crash the whole wizard. Non-interactive mode skips the loop —
            # bad values from .env are caught by the runtime guard in
            # run_infra_step and surfaced as WizardStepError.
            value = prompts.with_validation_loop(
                lambda msg=message, cur=current: prompts.ask_text(msg, default=cur) or cur,
                _validate_port,
            )
        else:
            value = prompts.ask_text(message, default=current)
        updates[key] = value or current
    return updates


def _persist_env(env_path: Path, existing: dict[str, str], updates: dict[str, str]) -> None:
    """Write ``.env`` only when the merged data actually differs from disk.

    When the values are unchanged we skip the atomic rewrite to avoid
    unnecessary mtime churn, but we still enforce ``0600`` on the existing
    file.  A user who manually copied ``.env.example`` with a permissive umask
    (e.g. ``0644``) and then re-runs init without changing any values would
    otherwise keep group/world-readable secrets.
    """
    merged = merge_env(existing, updates)
    if merged == existing:
        harden_env_file_mode(env_path)
        return
    atomic_write_env(env_path, merged)


def _ensure_ollama_models(host: str) -> list[str]:
    """Return the list of models that were actually pulled (possibly empty)."""
    installed = set(runners.ollama_list(host=host))
    missing = [m for m in REQUIRED_OLLAMA_MODELS if m not in installed]
    pulled: list[str] = []
    for model in missing:
        print(f"  • pulling {model} (this may take several minutes)…")
        runners.ollama_pull(model)
        pulled.append(model)
    return pulled


def run_infra_step(
    repo_root: Path,
    env_path: Path | None = None,
    *,
    ollama_host: str = "http://localhost:11434",
) -> None:
    """Drive the full infra sub-flow. Idempotent — re-running is safe."""
    env_file = env_path if env_path is not None else (repo_root / ".env")
    existing = load_env(env_file)
    updates = _prompt_pg_values(existing)
    _persist_env(env_file, existing, updates)

    print("  • starting Docker compose stack…")
    runners.docker_compose_up(repo_root, env_path=env_file)

    host = updates.get("POSTGRES_HOST", "localhost")
    raw_port = updates.get("POSTGRES_PORT", "5432")
    try:
        port = int(raw_port.strip() if isinstance(raw_port, str) else raw_port)
    except (ValueError, AttributeError) as exc:
        raise WizardStepError(
            f"invalid POSTGRES_PORT value: {raw_port!r}",
            hint="port must be a positive integer (default 5432)",
        ) from exc
    if not (1 <= port <= 65535):
        raise WizardStepError(
            f"POSTGRES_PORT out of range (1-65535): {port}",
            hint="set POSTGRES_PORT in .env to a value between 1 and 65535",
        )
    print(f"  • waiting for PostgreSQL on {host}:{port}…")
    runners.wait_pg_ready(host, port)

    print("  • applying Alembic migrations…")
    runners.alembic_upgrade_head(repo_root, env_path=env_file)

    print("  • checking Ollama models…")
    pulled = _ensure_ollama_models(ollama_host)
    if pulled:
        print(f"  • pulled: {', '.join(pulled)}")
    else:
        print("  • all required Ollama models already installed")


__all__ = ["REQUIRED_OLLAMA_MODELS", "run_infra_step"]
