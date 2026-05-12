"""Step 6: end-to-end sanity probes after the wizard has finished writing config.

Each probe is independent — we run all of them, collect failures, then print
a single structured table and return the failure count. The wizard caller
turns a non-zero count into a non-zero process exit code so CI / launchd
``RunAtLoad`` callers can detect a broken install.

The four probes are:

* DB ping (``SELECT 1`` via :func:`argos.init_wizard.runners.db_ping`).
* Ollama HTTP probe (``/api/tags``).
* Slack ``auth.test``.
* launchd ``is_loaded`` for both Argos plists (only when ARG-51 is present).
"""

from __future__ import annotations

from pathlib import Path

from argos import config_store
from argos.init_wizard import runners
from argos.init_wizard.env_file import load_env

try:
    from argos.scheduler import is_loaded  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - until ARG-51 merges
    is_loaded = None  # type: ignore[assignment]

# launchd labels we expect ARG-51 to install.
LAUNCHD_LABELS: tuple[str, ...] = ("com.argos.run", "com.argos.brief")


def _probe(
    name: str,
    fn,
    *args,
    **kwargs,
) -> tuple[str, str, str]:
    """Run ``fn(*args, **kwargs)`` and return a ``(name, status, detail)`` row."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any failure is a healthcheck failure
        return (name, "FAIL", str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
    return (name, "OK", "")


def _print_table(rows: list[tuple[str, str, str]]) -> None:
    name_w = max(len(r[0]) for r in rows)
    status_w = max(len(r[1]) for r in rows)
    for name, status, detail in rows:
        marker = "✓" if status == "OK" else "✗"
        line = f"  {marker} {name.ljust(name_w)}  {status.ljust(status_w)}"
        if detail:
            line += f"  — {detail}"
        print(line)


def run_healthcheck_step(
    repo_root: Path,
    env_path: Path | None = None,
    *,
    ollama_host: str = "http://localhost:11434",
) -> int:
    """Run every probe and return the number of failures (0 = healthy)."""
    env_file = env_path or config_store.default_env_path()
    env = load_env(env_file)
    bot_token = env.get("SLACK_BOT_TOKEN", "")

    rows: list[tuple[str, str, str]] = []

    rows.append(_probe("PostgreSQL", runners.run_async, runners.db_ping()))
    rows.append(_probe("Ollama", runners.ollama_ping, ollama_host))
    if bot_token:
        rows.append(_probe("Slack auth.test", runners.slack_auth_test, bot_token))
    else:
        rows.append(("Slack auth.test", "SKIP", "SLACK_BOT_TOKEN not set"))

    if is_loaded is None:
        for label in LAUNCHD_LABELS:
            rows.append((f"launchd {label}", "SKIP", "requires ARG-51 scheduler module"))
    else:
        for label in LAUNCHD_LABELS:
            def _check(_label: str = label) -> None:
                if not is_loaded(_label):  # type: ignore[misc]
                    raise RuntimeError(
                        "not loaded — are you in a graphical login session? "
                        "launchctl bootstrap requires the gui/<uid> domain"
                    )

            rows.append(_probe(f"launchd {label}", _check))

    _print_table(rows)
    failures = sum(1 for r in rows if r[1] == "FAIL")
    return failures


__all__ = ["LAUNCHD_LABELS", "run_healthcheck_step"]
