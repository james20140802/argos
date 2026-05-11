"""Step 1: confirm required binaries are installed before the wizard mutates anything.

We deliberately keep the precheck cheap: we only confirm ``docker`` and ``ollama``
are on ``PATH``. The infra step does deeper liveness probes (``docker info`` via
``docker compose up``, ``ollama list`` via HTTP). Catching missing binaries here
gives the user an actionable hint before anything else fires.
"""

from __future__ import annotations

from argos.init_wizard import WizardAbort
from argos.init_wizard import runners

# Each binary maps to a one-line install hint the user can copy/paste.
_REQUIRED_BINARIES: dict[str, str] = {
    "docker": (
        "install Docker Desktop for macOS: https://www.docker.com/products/docker-desktop "
        "(or `brew install --cask docker`)"
    ),
    "ollama": (
        "install Ollama: https://ollama.com/download (or `brew install ollama`)"
    ),
    "uv": (
        "install uv (required for migrations and CLI): https://github.com/astral-sh/uv "
        "(or `brew install uv`)"
    ),
}


def run_precheck_step() -> None:
    """Verify every required binary resolves on ``PATH``.

    Raises :class:`argos.init_wizard.WizardAbort` listing all missing binaries
    so the user can fix them in one pass rather than re-running the wizard
    once per missing tool.
    """
    missing: list[tuple[str, str]] = []
    for binary, hint in _REQUIRED_BINARIES.items():
        if runners.which(binary) is None:
            missing.append((binary, hint))
    if not missing:
        return
    lines = ["required binaries are not on PATH:"]
    for binary, hint in missing:
        lines.append(f"  - {binary}: {hint}")
    raise WizardAbort("\n".join(lines))


__all__ = ["run_precheck_step"]
