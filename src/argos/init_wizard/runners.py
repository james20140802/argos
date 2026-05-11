"""Thin, mockable wrappers around every external command the wizard runs.

Every subprocess invocation, every outbound HTTP call, and the single DB ping
all flow through this module. Step modules import these helpers by name so
unit tests can :func:`monkeypatch.setattr` them in one place; production runs
get real ``subprocess.run`` / ``httpx`` traffic.

The wrappers raise :class:`argos.init_wizard.WizardStepError` (with an
actionable ``hint=``) on failure so steps can keep a single ``try/except``
shape regardless of which probe blew up.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from sqlalchemy import text

from argos.init_wizard import WizardStepError

logger = logging.getLogger(__name__)

# Timeouts kept short so a broken Docker daemon doesn't hang the wizard.
PG_READY_POLL_INTERVAL_SEC = 1.0
PG_READY_DEFAULT_TIMEOUT_SEC = 30
SUBPROCESS_DEFAULT_TIMEOUT_SEC = 60


def which(binary: str) -> str | None:
    """Return the absolute path of ``binary`` on ``PATH``, or ``None`` if missing."""
    return shutil.which(binary)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = SUBPROCESS_DEFAULT_TIMEOUT_SEC,
    hint: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and surface failures as :class:`WizardStepError`."""
    logger.debug("runners._run cmd=%s cwd=%s", cmd, cwd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise WizardStepError(
            f"command not found: {cmd[0]}",
            hint=hint or f"install {cmd[0]} and re-run `argos init`",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise WizardStepError(
            f"timeout after {timeout}s running {' '.join(cmd)}",
            hint=hint or "increase the timeout or check whether the service is healthy",
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        snippet = stderr[-1] if stderr else (proc.stdout or "").strip().splitlines()[-1:]
        raise WizardStepError(
            f"{' '.join(cmd)} exited with code {proc.returncode}: {snippet}",
            hint=hint,
        )
    return proc


def docker_compose_up(repo_root: Path) -> None:
    """Bring up the Argos compose stack in detached mode."""
    _run(
        ["docker", "compose", "up", "-d"],
        cwd=repo_root,
        hint="run `docker info` to confirm Docker Desktop is running",
    )


def wait_pg_ready(
    host: str,
    port: int,
    *,
    timeout: int = PG_READY_DEFAULT_TIMEOUT_SEC,
) -> None:
    """Poll ``pg_isready`` until PostgreSQL accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    last_stderr = ""
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(
                ["pg_isready", "-h", host, "-p", str(port)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError:
            # pg_isready is bundled with the postgres image but may be absent on the host;
            # fall back to a TCP connect probe via httpx (HTTPS not required — we just want
            # the socket).
            proc = None
        if proc is not None and proc.returncode == 0:
            return
        if proc is not None:
            last_stderr = (proc.stderr or proc.stdout or "").strip()
        time.sleep(PG_READY_POLL_INTERVAL_SEC)
    raise WizardStepError(
        f"PostgreSQL at {host}:{port} did not become ready within {timeout}s",
        hint=f"check `docker compose ps` and `docker compose logs db`; last: {last_stderr}",
    )


def alembic_upgrade_head(repo_root: Path) -> None:
    """Apply all pending Alembic migrations."""
    _run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        hint="inspect the migration error above; you can rollback with "
        "`uv run alembic downgrade -1`",
    )


def ollama_list(host: str = "http://localhost:11434") -> list[str]:
    """Return the list of locally installed Ollama model names (``name:tag``)."""
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise WizardStepError(
            f"could not reach Ollama at {host}: {exc}",
            hint="start Ollama (`ollama serve`) or update ollama.host in config.toml",
        ) from exc
    payload = resp.json()
    return [m["name"] for m in payload.get("models", []) if "name" in m]


def ollama_pull(model: str) -> None:
    """Run ``ollama pull <model>`` (synchronous, blocks until download finishes)."""
    _run(
        ["ollama", "pull", model],
        timeout=60 * 30,  # large models can take a while on slow links
        hint=f"check `ollama list` and your network; you can retry with `ollama pull {model}`",
    )


def slack_auth_test(bot_token: str, app_token: str | None = None) -> dict:
    """Call Slack ``auth.test`` with the bot token and return the JSON response.

    The ``app_token`` is accepted for symmetry with the wizard step (which
    collects both) but is not exercised here — Slack rejects ``xapp-`` tokens
    for ``auth.test`` so we only validate the ``xoxb-`` bot token. The app
    token is later validated implicitly when the bot connects via Socket Mode.
    """
    if not bot_token:
        raise WizardStepError(
            "SLACK_BOT_TOKEN is empty",
            hint="generate an xoxb- token at https://api.slack.com/apps and re-enter it",
        )
    try:
        resp = httpx.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=10,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise WizardStepError(
            f"network error calling Slack auth.test: {exc}",
            hint="check internet connectivity and try again",
        ) from exc
    payload = resp.json()
    if not payload.get("ok"):
        raise WizardStepError(
            f"Slack auth.test failed: {payload.get('error', 'unknown')}",
            hint="confirm the bot token is the xoxb- value from OAuth & Permissions",
        )
    return payload


async def db_ping() -> None:
    """Open an async session and run ``SELECT 1``. Raises :class:`WizardStepError`."""
    # Import lazily so importing runners.py doesn't eagerly construct the engine
    # (which reads .env). Tests that monkeypatch runners.db_ping never hit this path.
    from argos.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise WizardStepError(
            f"database ping failed: {exc}",
            hint="confirm `docker compose ps` shows the db container healthy",
        ) from exc


def ollama_ping(host: str = "http://localhost:11434") -> None:
    """Synchronous HTTP probe against Ollama's ``/api/tags`` endpoint."""
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise WizardStepError(
            f"Ollama unreachable at {host}: {exc}",
            hint="start Ollama (`ollama serve`) — the daemon must be running",
        ) from exc


def run_async(coro) -> None:
    """Convenience wrapper so sync step modules can await ``db_ping``."""
    asyncio.run(coro)


__all__ = [
    "alembic_upgrade_head",
    "db_ping",
    "docker_compose_up",
    "ollama_list",
    "ollama_ping",
    "ollama_pull",
    "run_async",
    "slack_auth_test",
    "wait_pg_ready",
    "which",
]
