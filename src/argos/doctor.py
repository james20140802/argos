"""Read-only health probes for `argos doctor`.

Each function returns a ``(name, status, detail)`` tuple where ``status`` is
one of ``"OK"``, ``"FAIL"``, or ``"WARN"``.  ``"WARN"`` rows do not increment
the failure count in ``_cmd_doctor``; ``"FAIL"`` rows do.

Probes reuse ``argos.init_wizard.runners`` helpers (``which``, ``ollama_list``)
so external-call logic stays in one place.
"""

from __future__ import annotations

import platform
import subprocess
import sys

Row = tuple[str, str, str]  # (name, status, detail)


def check_docker() -> Row:
    """Probe: Docker daemon is reachable.

    1. Look for the ``docker`` binary via ``runners.which``.
    2. Run ``docker info`` with a 5-second timeout.
    """
    from argos.init_wizard import runners

    if runners.which("docker") is None:
        return ("Docker daemon", "FAIL", "docker binary not found — install Docker Desktop or Colima")

    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ("Docker daemon", "FAIL", "docker info timed out — is Docker daemon running?")
    except FileNotFoundError:
        return ("Docker daemon", "FAIL", "docker binary not found — install Docker Desktop or Colima")

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        short = detail[-1] if detail else "non-zero exit"
        return ("Docker daemon", "FAIL", short)

    return ("Docker daemon", "OK", "")


def check_ollama_installed() -> Row:
    """Probe: ollama binary is on PATH."""
    from argos.init_wizard import runners

    if runners.which("ollama") is None:
        return ("Ollama installed", "FAIL", "ollama binary not found — install from https://ollama.com")
    return ("Ollama installed", "OK", "")


def check_ollama_qwen3_8b(ollama_host: str = "http://localhost:11434") -> Row:
    """Probe: qwen3:8b model is pulled locally.

    Calls ``runners.ollama_list()`` against *ollama_host* and looks for an
    exact match on ``qwen3:8b``.  Converts ``WizardStepError`` (Ollama
    unreachable) to FAIL.

    Args:
        ollama_host: Base URL for the Ollama API (e.g. ``http://localhost:11434``).
            Defaults to the standard local address; callers should pass
            ``cfg.ollama.host`` so the probe honours the configured host.
    """
    from argos.init_wizard import runners
    from argos.init_wizard import WizardStepError

    try:
        models = runners.ollama_list(host=ollama_host)
    except WizardStepError as exc:
        return ("Qwen3-8B pulled", "FAIL", str(exc).splitlines()[0])

    if "qwen3:8b" in models:
        return ("Qwen3-8B pulled", "OK", "")

    return ("Qwen3-8B pulled", "FAIL", "model not found — run: ollama pull qwen3:8b")


def check_ollama_models(ollama_host: str = "http://localhost:11434") -> list[Row]:
    """Probe all Ollama models required by Argos.

    Returns one ``Row`` per required model (``qwen3:8b``, ``qwen3:32b``,
    ``nomic-embed-text``).  A single ``ollama list`` call is made; if Ollama is
    unreachable all rows are marked FAIL with the same error detail.

    Args:
        ollama_host: Base URL for the Ollama API.  Callers should pass
            ``cfg.ollama.host`` so the probe honours the configured host.
    """
    from argos.init_wizard import runners
    from argos.init_wizard import WizardStepError
    from argos.init_wizard.steps.infra import REQUIRED_OLLAMA_MODELS

    try:
        available = runners.ollama_list(host=ollama_host)
    except WizardStepError as exc:
        error_detail = str(exc).splitlines()[0]
        return [(model, "FAIL", error_detail) for model in REQUIRED_OLLAMA_MODELS]

    rows: list[Row] = []
    for model in REQUIRED_OLLAMA_MODELS:
        # Use exact match to align with the runtime pull check in infra.py
        # (_ensure_ollama_models), which also uses exact membership.  Prefix
        # matching was previously used here but caused false-OK results when a
        # differently-tagged variant (e.g. qwen3:8b-instruct) was installed
        # instead of the exact model ID that inference calls request.
        if model in available:
            rows.append((model, "OK", ""))
        else:
            rows.append((model, "FAIL", f"model not found — run: ollama pull {model}"))
    return rows


def check_uv_installed() -> Row:
    """Probe: uv binary is on PATH.

    ``argos init`` hard-requires ``uv`` (``_REQUIRED_BINARIES`` in
    ``init_wizard/steps/precheck.py``) and migrations are executed via
    ``uv run alembic``.  A missing ``uv`` makes the next init step fail
    immediately, so this probe surfaces the gap at doctor time.
    """
    from argos.init_wizard import runners

    if runners.which("uv") is None:
        return (
            "uv installed",
            "FAIL",
            "uv binary not found — install from https://github.com/astral-sh/uv (or `brew install uv`)",
        )
    return ("uv installed", "OK", "")


def check_python_version() -> Row:
    """Probe: Python version is >=3.10 and <3.13."""
    vi = sys.version_info
    # Use index access so tests can patch sys.version_info with a plain tuple.
    version_str = f"{vi[0]}.{vi[1]}.{vi[2]}"

    if vi < (3, 10):
        return ("Python version", "FAIL", f"{version_str} — requires >=3.10")
    if vi >= (3, 13):
        return ("Python version", "FAIL", f"{version_str} — requires <3.13")
    return ("Python version", "OK", version_str)


def check_macos_version() -> Row:
    """Probe: macOS major version is >=12 (Monterey).

    This is a soft WARN, not FAIL, because older macOS may still work but is
    untested.  Non-macOS hosts always pass so Linux CI is not broken.
    """
    mac_ver, _, _ = platform.mac_ver()
    if not mac_ver:
        # Non-macOS — not a requirement, skip gracefully.
        return ("macOS version", "OK", f"{platform.system()} — macOS check skipped")

    try:
        major = int(mac_ver.split(".")[0])
    except (ValueError, IndexError):
        return ("macOS version", "WARN", f"could not parse macOS version: {mac_ver!r}")

    if major < 12:
        return ("macOS version", "WARN", f"{mac_ver} — Monterey (12) or later recommended")
    return ("macOS version", "OK", mac_ver)


def print_doctor_table(rows: list[Row]) -> None:
    """Print a structured table of probe results to stdout."""
    if not rows:
        return
    name_w = max(len(r[0]) for r in rows)
    status_w = max(len(r[1]) for r in rows)
    print("\nargos doctor")
    print("─" * (name_w + status_w + 12))
    for name, status, detail in rows:
        if status == "OK":
            marker = "✓"
        elif status == "WARN":
            marker = "!"
        else:
            marker = "✗"
        line = f"  {marker} {name.ljust(name_w)}  {status.ljust(status_w)}"
        if detail:
            line += f"  — {detail}"
        print(line)
    print()


__all__ = [
    "check_docker",
    "check_macos_version",
    "check_ollama_installed",
    "check_ollama_models",
    "check_ollama_qwen3_8b",
    "check_python_version",
    "check_uv_installed",
    "print_doctor_table",
]
