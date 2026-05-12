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
import os
import shutil
import socket
import subprocess
import sys
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
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and surface failures as :class:`WizardStepError`.

    ``env``, when provided, is passed verbatim to :func:`subprocess.run` as
    the child process environment.  Callers that need to inject extra
    environment variables while preserving the current process's env should
    pass ``{**os.environ, ...}``.
    """
    logger.debug("runners._run cmd=%s cwd=%s", cmd, cwd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
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


def docker_compose_up(repo_root: Path, env_path: Path | None = None) -> None:
    """Bring up the Argos compose stack in detached mode.

    If ``env_path`` is supplied it is passed to ``docker compose`` via the
    ``--env-file`` flag so the compose stack reads Postgres credentials from the
    same ``.env`` file that :func:`wait_pg_ready` and
    :func:`alembic_upgrade_head` use.  Without this, Docker would fall back to
    its own ``.env`` search rules which may point at a different file when the
    caller passed a non-default path.
    """
    cmd = ["docker", "compose"]
    if env_path is not None:
        cmd += ["--env-file", str(env_path)]
    cmd += ["up", "-d"]
    _run(
        cmd,
        cwd=repo_root,
        hint="run `docker info` to confirm Docker Desktop is running",
    )


def _socket_probe(host: str, port: int, *, timeout: float = 2.0) -> bool:
    """Best-effort TCP-connect readiness check used as the ``pg_isready`` fallback.

    ``pg_isready`` is the preferred probe because it understands the Postgres
    startup handshake; but on a clean machine that has Docker + Ollama but no
    local Postgres client binaries the executable is missing. A bare TCP
    connect is sufficient for our purposes: the Postgres server only opens its
    listening socket once it's ready to accept startup packets, so a successful
    ``connect()`` is a strong proxy for "ready". Returns ``False`` on any
    socket error (refused / unreachable / DNS / timeout).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_pg_ready(
    host: str,
    port: int,
    *,
    timeout: int = PG_READY_DEFAULT_TIMEOUT_SEC,
) -> None:
    """Poll ``pg_isready`` until PostgreSQL accepts connections or timeout expires.

    Falls back to a TCP ``connect()`` probe when the ``pg_isready`` binary is
    absent from the host (common on clean machines that have Docker + Ollama
    but no local Postgres client tools installed).
    """
    deadline = time.monotonic() + timeout
    last_stderr = ""
    use_socket_fallback = False
    while time.monotonic() < deadline:
        if use_socket_fallback:
            # pg_isready missing: drop straight into the socket probe path.
            remaining = max(0.1, min(2.0, deadline - time.monotonic()))
            if _socket_probe(host, port, timeout=remaining):
                return
            last_stderr = "pg_isready unavailable; TCP connect refused"
            time.sleep(PG_READY_POLL_INTERVAL_SEC)
            continue
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
            # switch to a TCP connect probe for the rest of the polling window.
            use_socket_fallback = True
            continue
        if proc.returncode == 0:
            return
        last_stderr = (proc.stderr or proc.stdout or "").strip()
        time.sleep(PG_READY_POLL_INTERVAL_SEC)
    raise WizardStepError(
        f"PostgreSQL at {host}:{port} did not become ready within {timeout}s",
        hint=f"check `docker compose ps` and `docker compose logs db`; last: {last_stderr}",
    )


def alembic_upgrade_head(repo_root: Path, env_path: Path | None = None) -> None:
    """Apply all pending Alembic migrations.

    If ``env_path`` is supplied its ``POSTGRES_*`` (and other) key/value pairs
    are merged into the subprocess environment so Alembic's ``argos.config.Secrets``
    reads the correct credentials — even when the caller used a non-default
    ``.env`` location that differs from the working directory.
    """
    extra_env: dict[str, str] | None = None
    if env_path is not None:
        from argos.init_wizard.env_file import load_env

        extra_env = {**os.environ, **load_env(env_path)}

    _run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        hint="inspect the migration error above; you can rollback with "
        "`uv run alembic downgrade -1`",
        env=extra_env,
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
    token is validated separately via :func:`slack_app_connections_open`.
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


def slack_app_connections_open(app_token: str) -> dict:
    """Call Slack ``apps.connections.open`` with the app token and return the JSON response.

    This validates the ``xapp-`` app-level token before Socket Mode is started,
    catching revoked, foreign, or typoed tokens at wizard time rather than at
    bot startup. The endpoint is the exact one used by the Socket Mode client
    internally, so a successful probe is strong evidence the connection will work.
    """
    if not app_token:
        raise WizardStepError(
            "SLACK_APP_TOKEN is empty",
            hint="generate an xapp- token at https://api.slack.com/apps (Socket Mode) and re-enter it",
        )
    try:
        resp = httpx.post(
            "https://slack.com/api/apps.connections.open",
            headers={"Authorization": f"Bearer {app_token}"},
            timeout=10,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise WizardStepError(
            f"network error calling Slack apps.connections.open: {exc}",
            hint="check internet connectivity and try again",
        ) from exc
    payload = resp.json()
    if not payload.get("ok"):
        raise WizardStepError(
            "slack app token rejected by apps.connections.open",
            hint=payload.get("error", "unknown"),
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


# ---------------------------------------------------------------------------
# Playwright Chromium helpers
# ---------------------------------------------------------------------------

# 30 minutes mirrors the ollama_pull timeout — the ~150MB download can be slow.
PLAYWRIGHT_INSTALL_TIMEOUT_SEC = 60 * 30


def _run_streaming(
    cmd: list[str],
    *,
    timeout: int = PLAYWRIGHT_INSTALL_TIMEOUT_SEC,
    hint: str | None = None,
) -> None:
    """Run ``cmd`` with stdout/stderr streamed live to the terminal.

    Unlike :func:`_run`, output is not captured so the user can see progress
    bars from long-running commands (e.g. ``playwright install chromium``).
    On non-zero exit a :class:`WizardStepError` is raised with ``hint``; the
    user already saw any error output on screen, so we omit the snippet.
    """
    logger.debug("runners._run_streaming cmd=%s", cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=False,
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
        raise WizardStepError(
            f"{' '.join(cmd)} exited with code {proc.returncode}",
            hint=hint,
        )


def playwright_chromium_installed() -> bool:
    """Return ``True`` if Playwright's Chromium executable exists on disk.

    Uses Playwright's own Python API to resolve the expected executable path,
    then checks for its presence with :func:`os.path.exists`. This is faster
    and more reliable than parsing ``playwright install --dry-run`` output and
    avoids spawning Node. Returns ``False`` on any Playwright ``Error`` (e.g.
    the browser has never been installed) or if the resolved path is absent.
    """
    try:
        # Lazy import so wizard startup is not slowed by the Playwright import
        # chain when this module is loaded but the helper is not yet called.
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return os.path.exists(path)
    except Exception:  # noqa: BLE001 — covers playwright Error + any import issue
        return False


def playwright_install_chromium() -> None:
    """Run ``playwright install chromium`` and stream progress to the terminal.

    Invokes Playwright via the current Python interpreter
    (``sys.executable -m playwright install chromium``) rather than a bare
    ``playwright`` entry point.  This is necessary because pipx only exposes
    the package's own entry points; dependency apps such as ``playwright`` are
    not on PATH unless ``--include-deps`` was used.  Using ``sys.executable``
    guarantees the call lands in the same venv that Argos itself is running in,
    regardless of installation method.  Raises :class:`WizardStepError` with a
    manual-fallback hint on non-zero exit.
    """
    _run_streaming(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        timeout=PLAYWRIGHT_INSTALL_TIMEOUT_SEC,
        hint="run `python -m playwright install chromium` manually and then re-run `argos init`",
    )


__all__ = [
    "alembic_upgrade_head",
    "db_ping",
    "docker_compose_up",
    "ollama_list",
    "ollama_ping",
    "ollama_pull",
    "playwright_chromium_installed",
    "playwright_install_chromium",
    "run_async",
    "slack_app_connections_open",
    "slack_auth_test",
    "wait_pg_ready",
    "which",
]
