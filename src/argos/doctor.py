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


def check_postgres_reachable() -> Row:
    """Probe: Postgres is reachable (runs SELECT 1 via the async engine).

    Reuses ``runners.db_ping`` so the DB-connection logic lives in one place.
    A ``WizardStepError`` (ping raised) becomes a FAIL row with its message.
    """
    from argos.init_wizard import runners
    from argos.init_wizard import WizardStepError

    try:
        runners.run_async(runners.db_ping())
    except WizardStepError as exc:
        return ("Postgres reachable", "FAIL", str(exc).splitlines()[0])
    except Exception as exc:  # pragma: no cover - defensive
        return ("Postgres reachable", "FAIL", f"unexpected error: {exc}")
    return ("Postgres reachable", "OK", "")


def _alembic_current_and_head() -> tuple[str | None, str | None]:
    """Return (current_db_revision, script_head_revision), read-only.

    ``head`` comes from the migration scripts (no DB needed).  ``current`` is
    read from the ``alembic_version`` table via the async engine.  Multi-head
    repos are not expected here; the first head is used.
    """
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    from argos.database import AsyncSessionLocal
    from argos.init_wizard import runners

    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    head = ScriptDirectory.from_config(cfg).get_current_head()

    current: str | None = None

    async def _read_current() -> None:
        nonlocal current
        async with AsyncSessionLocal() as session:
            row = await session.execute(text("SELECT version_num FROM alembic_version"))
            current = row.scalar_one_or_none()

    runners.run_async(_read_current())
    return current, head


def check_alembic_head() -> Row:
    """Probe: applied DB revision equals the latest migration head."""
    try:
        current, head = _alembic_current_and_head()
    except Exception as exc:
        return ("Alembic migrations", "FAIL", f"could not determine revision: {exc}")

    if current is None:
        return ("Alembic migrations", "FAIL", "no alembic_version row — run: uv run alembic upgrade head")
    if current != head:
        return (
            "Alembic migrations",
            "FAIL",
            f"current {current} != head {head} — run: uv run alembic upgrade head",
        )
    return ("Alembic migrations", "OK", current)


_VRAM_WARN_THRESHOLD_BYTES = 4 * 1024**3  # 4 GiB free-memory floor (advisory)


def _available_memory_bytes() -> int | None:
    """Best-effort available system memory in bytes (macOS via vm_stat/sysctl).

    Returns None when it cannot be determined (non-macOS or parse failure) so
    the caller degrades to a WARN rather than crashing.  Uses only stdlib +
    subprocess — no third-party dependency.
    """
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None

    page_size = 4096
    free_pages = 0
    for line in vm.stdout.splitlines():
        if "page size of" in line:
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                page_size = int(digits)
        elif line.startswith("Pages free:") or line.startswith("Pages inactive:"):
            digits = "".join(ch for ch in line.split(":", 1)[1] if ch.isdigit())
            if digits:
                free_pages += int(digits)
    if free_pages == 0:
        return None
    return free_pages * page_size


def _loaded_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """Model names currently loaded in Ollama (via ``/api/ps``).

    Raises on transport error; the caller converts that to a WARN.
    """
    import httpx

    resp = httpx.get(f"{host.rstrip('/')}/api/ps", timeout=5)
    resp.raise_for_status()
    return [m.get("name", "") for m in resp.json().get("models", [])]


def check_vram_headroom(ollama_host: str = "http://localhost:11434") -> Row:
    """Probe: enough free unified memory to load a model without pressure.

    Advisory (never FAIL): reports loaded models + free GiB, WARNs when free
    memory is below the threshold or when either input is unavailable.
    """
    try:
        loaded = _loaded_ollama_models(ollama_host)
    except Exception as exc:
        return ("VRAM headroom", "WARN", f"Ollama /api/ps unreachable: {exc}")

    free = _available_memory_bytes()
    if free is None:
        loaded_str = ", ".join(loaded) if loaded else "none"
        return ("VRAM headroom", "WARN", f"could not read free memory (loaded: {loaded_str})")

    free_gib = free / 1024**3
    loaded_str = ", ".join(loaded) if loaded else "none loaded"
    detail = f"{free_gib:.1f} GiB free (loaded: {loaded_str})"
    if free < _VRAM_WARN_THRESHOLD_BYTES:
        return ("VRAM headroom", "WARN", detail + " — low headroom")
    return ("VRAM headroom", "OK", detail)


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
    "check_alembic_head",
    "check_docker",
    "check_macos_version",
    "check_ollama_installed",
    "check_ollama_models",
    "check_ollama_qwen3_8b",
    "check_postgres_reachable",
    "check_python_version",
    "check_uv_installed",
    "check_vram_headroom",
    "print_doctor_table",
]
