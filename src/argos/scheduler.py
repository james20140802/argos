"""launchd plist rendering + install/bootstrap helpers (ARG-51).

This module owns the launchd surface for Argos. It renders plist XML
via stdlib :mod:`plistlib` (never hand-crafted strings), atomically
installs the file via ``write→os.replace``, and shells out to
``launchctl bootstrap`` / ``bootout`` / ``print`` through a single
centralized helper so tests can swap subprocess behavior with a
monkeypatch.

Public API (the surface ARG-48's ``argos init`` imports):

* :func:`render_run_plist`
* :func:`render_brief_plist`
* :func:`install_plist`
* :func:`bootstrap_plist`
* :func:`bootout_plist`
* :func:`is_loaded`
* :func:`reload_schedule`

launchd ``Weekday`` integer convention is **Sun=0, Mon=1, …, Sat=6**,
NOT ISO 1..7. Picking the wrong mapping silently shifts the briefing
by one day — see :func:`_weekday_to_launchd` and its unit tests.
"""

from __future__ import annotations

import logging
import os
import plistlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SchedulerError(RuntimeError):
    """Raised when a launchd interaction fails or the environment is misconfigured."""


# launchd's StartCalendarInterval Weekday key uses Sun=0..Sat=6 (NOT ISO).
# Documented at: man launchd.plist → StartCalendarInterval / Weekday.
_WEEKDAY_TO_LAUNCHD: dict[str, int] = {
    "Sun": 0,
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
}

_DEFAULT_LOG_DIR = Path.home() / "Library" / "Logs" / "argos"
_DEFAULT_LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
# Apple Silicon Homebrew (/opt/homebrew) is listed first so that M1 Max
# binaries (Argos's primary target) take precedence in launchd's minimal
# PATH. Intel Homebrew (/usr/local/bin) and POSIX fallbacks follow so the
# plist also works on x86 Macs. launchd tolerates non-existent PATH entries,
# so the extra segments are harmless on machines where /opt/homebrew doesn't
# exist.
_DEFAULT_ENV_PATH = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin"

# Fallback locations to search if `shutil.which("argos")` returns None.
# Order matters: Argos's primary target is Apple Silicon (M1 Max), where
# Homebrew installs to /opt/homebrew/bin. GUI launches and launchd's own
# spawn context have a minimal PATH, so `shutil.which` can return None
# even when `argos` is installed — we must probe the well-known Homebrew
# locations before giving up.
_ARGOS_BINARY_FALLBACKS: tuple[Path, ...] = (
    Path("/opt/homebrew/bin/argos"),  # Apple Silicon Homebrew (primary target)
    Path("/usr/local/bin/argos"),  # Intel Homebrew / generic Unix
    Path.home() / ".local" / "bin" / "argos",  # user pip/uv site
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_hhmm(s: str) -> tuple[int, int]:
    """Parse ``HH:MM`` (24h) into ``(hour, minute)``. Strict — single-digit
    hour is allowed (``6:00``) but minute must be two digits and both ranges
    must be valid clock time. Raises :class:`ValueError` on anything else.
    """
    if not isinstance(s, str):
        raise ValueError(f"time must be a string, got {type(s).__name__}")
    if ":" not in s:
        raise ValueError(f"time {s!r} must be in HH:MM format")
    hh, _, mm = s.partition(":")
    if not hh.isdigit() or not mm.isdigit():
        raise ValueError(f"time {s!r} must contain only digits and ':'")
    if len(mm) != 2:
        raise ValueError(f"time {s!r} must have a two-digit minute component")
    hour = int(hh)
    minute = int(mm)
    if not (0 <= hour <= 23):
        raise ValueError(f"hour out of range in {s!r} (0-23)")
    if not (0 <= minute <= 59):
        raise ValueError(f"minute out of range in {s!r} (0-59)")
    return hour, minute


def _weekday_to_launchd(name: str) -> int:
    """Map a 3-letter day-name to launchd's Weekday integer (Sun=0..Sat=6)."""
    key = name.strip().capitalize()[:3]
    if key not in _WEEKDAY_TO_LAUNCHD:
        raise ValueError(
            f"unknown weekday {name!r}; expected one of {sorted(_WEEKDAY_TO_LAUNCHD)}"
        )
    return _WEEKDAY_TO_LAUNCHD[key]


def _calendar_intervals(
    hour: int, minute: int, weekdays: list[str] | None
) -> dict[str, int] | list[dict[str, int]]:
    """Build the ``StartCalendarInterval`` value.

    * ``None`` or a full 7-day list → single dict with just ``Hour``/``Minute``
      (launchd interprets the absent Weekday as "every day").
    * Subset → list-of-dicts, each carrying a ``Weekday`` int.

    Raises :class:`ValueError` if ``weekdays`` is an empty list — launchd
    interprets ``StartCalendarInterval: <array/>`` as "no schedule", so an
    empty weekdays list would silently disable the briefing job.
    """
    if weekdays is not None and len(weekdays) == 0:
        raise ValueError(
            "weekdays must be non-empty; set briefing.weekdays = ['Mon', 'Tue', ...]"
            " or omit to schedule daily"
        )
    base = {"Hour": hour, "Minute": minute}
    if weekdays is None or len(set(_weekday_to_launchd(w) for w in weekdays)) == 7:
        return base
    intervals: list[dict[str, int]] = []
    for name in weekdays:
        wd = _weekday_to_launchd(name)
        intervals.append({"Hour": hour, "Minute": minute, "Weekday": wd})
    return intervals


def _resolve_argos_binary() -> Path:
    """Locate the ``argos`` executable as an absolute path.

    launchd does NOT inherit the user's PATH, so the plist must embed an
    absolute path. Resolution order:

    1. ``shutil.which("argos")`` — honours the caller's PATH
    2. ``/opt/homebrew/bin/argos`` — Apple Silicon Homebrew default
       (Argos's primary target is M1 Max, so this is checked first)
    3. ``/usr/local/bin/argos`` — Intel Homebrew / generic Unix
    4. ``~/.local/bin/argos`` — user pip/uv site

    Each fallback is accepted only if ``Path.is_file()`` is true so we
    don't embed a stale directory or symlink-to-nowhere into the plist.
    Raises :class:`SchedulerError` with an operator-friendly message
    listing every probed location if none of the candidates exist.
    """
    found = shutil.which("argos")
    if found:
        return Path(found).resolve()
    for candidate in _ARGOS_BINARY_FALLBACKS:
        if candidate.is_file():
            return candidate.resolve()
    raise SchedulerError(
        "Could not locate the `argos` executable. Tried `shutil.which('argos')` "
        f"and fallbacks {[str(p) for p in _ARGOS_BINARY_FALLBACKS]!r}. "
        "Install argos to one of these locations or add it to PATH before "
        "scheduling."
    )


def _current_uid() -> int:
    return os.getuid()


def _run_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Centralized ``launchctl`` invocation. Tests monkeypatch this symbol."""
    cmd = ["launchctl", *args]
    logger.debug("launchctl invocation: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def _build_plist_dict(
    *,
    label: str,
    program_args: list[str],
    calendar: dict[str, int] | list[dict[str, int]],
    log_dir: Path,
    log_basename: str,
    working_directory: Path,
) -> dict[str, Any]:
    """Assemble the dict that :func:`plistlib.dumps` will serialize.

    ``WorkingDirectory`` is required because pydantic-settings resolves
    ``env_file=".env"`` against the process cwd — launchd otherwise runs
    jobs from ``/`` and silently falls back to empty config, which surfaces
    as DB auth failures at runtime.
    """
    return {
        "Label": label,
        "ProgramArguments": program_args,
        "StartCalendarInterval": calendar,
        "StandardOutPath": str(log_dir / f"{log_basename}.log"),
        "StandardErrorPath": str(log_dir / f"{log_basename}.log"),
        "EnvironmentVariables": {"PATH": _DEFAULT_ENV_PATH},
        "WorkingDirectory": str(working_directory),
        "RunAtLoad": False,
        "KeepAlive": False,
    }


def _render_plist(
    *,
    label: str,
    subcommand: str,
    log_basename: str,
    time: str,
    weekdays: list[str] | None,
    config_path: Path | None,
    log_dir: Path | None,
    working_directory: Path | None,
) -> str:
    hour, minute = _parse_hhmm(time)
    argos_bin = _resolve_argos_binary()
    program_args: list[str] = [str(argos_bin), subcommand]
    if config_path is not None:
        program_args.extend(["--config", str(config_path)])
    calendar = _calendar_intervals(hour, minute, weekdays)
    resolved_log_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
    resolved_cwd = (
        working_directory if working_directory is not None else Path.cwd()
    )
    payload = _build_plist_dict(
        label=label,
        program_args=program_args,
        calendar=calendar,
        log_dir=resolved_log_dir,
        log_basename=log_basename,
        working_directory=resolved_cwd,
    )
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML).decode("utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_run_plist(
    *,
    time: str,
    label: str = "com.argos.run",
    config_path: Path | None = None,
    log_dir: Path | None = None,
    working_directory: Path | None = None,
) -> str:
    """Render the ``argos run`` plist as XML text.

    ``working_directory`` is emitted as the plist's ``WorkingDirectory``
    so pydantic-settings can find ``.env`` at runtime. Defaults to
    ``Path.cwd()`` when not provided.
    """
    return _render_plist(
        label=label,
        subcommand="run",
        log_basename="run",
        time=time,
        weekdays=None,  # `argos run` is daily.
        config_path=config_path,
        log_dir=log_dir,
        working_directory=working_directory,
    )


def render_brief_weekly_plist(
    *,
    time: str,
    weekday: str = "Mon",
    label: str = "com.argos.brief-weekly",
    config_path: Path | None = None,
    log_dir: Path | None = None,
    working_directory: Path | None = None,
) -> str:
    """Render the ``argos brief --weekly`` plist as XML text (ARG-124).

    Schedules a single weekly run at ``time`` on ``weekday`` (3-letter
    day-name; mapped to Sun=0..Sat=6 via :func:`_weekday_to_launchd`).
    ``ProgramArguments`` includes the ``--weekly`` flag so the scheduled
    invocation dispatches the Keep portfolio summary instead of the daily
    briefing.

    Log path is ``brief-weekly.log`` under ``log_dir`` (or
    ``~/Library/Logs/argos`` by default).
    """
    hour, minute = _parse_hhmm(time)
    argos_bin = _resolve_argos_binary()
    program_args: list[str] = [str(argos_bin), "brief", "--weekly"]
    if config_path is not None:
        program_args.extend(["--config", str(config_path)])
    # Weekly = one launch per week → list-of-dicts with a single Weekday entry.
    wd = _weekday_to_launchd(weekday)
    calendar: list[dict[str, int]] = [
        {"Hour": hour, "Minute": minute, "Weekday": wd}
    ]
    resolved_log_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
    resolved_cwd = (
        working_directory if working_directory is not None else Path.cwd()
    )
    payload = _build_plist_dict(
        label=label,
        program_args=program_args,
        calendar=calendar,
        log_dir=resolved_log_dir,
        log_basename="brief-weekly",
        working_directory=resolved_cwd,
    )
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML).decode("utf-8")


def render_brief_plist(
    *,
    time: str,
    weekdays: list[str] | None = None,
    label: str = "com.argos.brief",
    config_path: Path | None = None,
    log_dir: Path | None = None,
    working_directory: Path | None = None,
) -> str:
    """Render the ``argos brief`` plist as XML text.

    ``weekdays`` accepts day-names (``["Mon", "Tue", …]``) to match
    :class:`argos.config.BriefingConfig.weekdays`. ``None`` or a full
    7-day list collapses to a daily-style ``StartCalendarInterval``.

    ``working_directory`` is emitted as the plist's ``WorkingDirectory``
    so pydantic-settings can find ``.env`` at runtime. Defaults to
    ``Path.cwd()`` when not provided.
    """
    return _render_plist(
        label=label,
        subcommand="brief",
        log_basename="brief",
        time=time,
        weekdays=weekdays,
        config_path=config_path,
        log_dir=log_dir,
        working_directory=working_directory,
    )


def render_web_plist(
    *,
    label: str = "com.argos.web",
    config_path: Path | None = None,
    log_dir: Path | None = None,
    working_directory: Path | None = None,
) -> str:
    """Render the persistent ``argos web`` daemon plist as XML text (ARG-165).

    Unlike the run/brief/brief-weekly schedules, this plist runs at login
    and is kept alive by launchd — there is no ``StartCalendarInterval``.
    """
    argos_bin = _resolve_argos_binary()
    program_args: list[str] = [str(argos_bin), "web"]
    if config_path is not None:
        program_args.extend(["--config", str(config_path)])
    resolved_log_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
    resolved_cwd = (
        working_directory if working_directory is not None else Path.cwd()
    )
    # Hand-build the payload so we can set RunAtLoad/KeepAlive=True and
    # omit StartCalendarInterval entirely — _build_plist_dict bakes a
    # scheduled-job shape that doesn't fit a long-running daemon.
    payload = {
        "Label": label,
        "ProgramArguments": program_args,
        "StandardOutPath": str(resolved_log_dir / "web.log"),
        "StandardErrorPath": str(resolved_log_dir / "web.log"),
        "EnvironmentVariables": {"PATH": _DEFAULT_ENV_PATH},
        "WorkingDirectory": str(resolved_cwd),
        "RunAtLoad": True,
        "KeepAlive": True,
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML).decode("utf-8")


def install_plist(plist_path: Path, content: str) -> None:
    """Write ``content`` to ``plist_path`` atomically with mode 0o644.

    Ensures the parent directory and the embedded log dir (under
    ``~/Library/Logs/argos`` unless overridden) exist before writing.
    On exception the temp file is cleaned up and the original file
    (if any) is left untouched.
    """
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    # Best-effort log dir creation: parse the plist for StandardOutPath
    # and ensure its parent exists. launchd will NOT create this directory.
    try:
        parsed = plistlib.loads(content.encode("utf-8"))
        out_path = parsed.get("StandardOutPath")
        if isinstance(out_path, str):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    except plistlib.InvalidFileException as exc:  # pragma: no cover - defensive
        raise SchedulerError(f"refusing to install malformed plist: {exc}") from exc

    tmp = plist_path.with_suffix(plist_path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(tmp, 0o644)
        os.replace(tmp, plist_path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _label_from_plist(plist_path: Path) -> str:
    with open(plist_path, "rb") as f:
        data = plistlib.load(f)
    label = data.get("Label")
    if not isinstance(label, str) or not label:
        raise SchedulerError(f"plist at {plist_path} is missing a Label key")
    return label


def is_loaded(label: str) -> bool:
    """Return True if a launchd job named ``label`` is loaded for the current GUI session.

    ``launchctl print`` output is not a stable API — we only check whether
    the command succeeded AND the label appears in stdout (lenient substring
    match per macOS-version risk noted in the plan).
    """
    target = f"gui/{_current_uid()}/{label}"
    result = _run_launchctl(["print", target])
    if result.returncode != 0:
        return False
    return label in (result.stdout or "")


def bootstrap_plist(plist_path: Path) -> None:
    """Load the plist into the current GUI session, replacing any stale copy.

    Idempotent: if the label is already loaded we ``bootout`` first
    (tolerating "not loaded" errors) and then re-``bootstrap``. Raises
    :class:`SchedulerError` if the final bootstrap fails.
    """
    if not plist_path.exists():
        raise SchedulerError(f"plist not found at {plist_path}")
    label = _label_from_plist(plist_path)
    domain = f"gui/{_current_uid()}"

    # Best-effort bootout to clear stale state. Non-zero is fine here (e.g.
    # exit 36 / "Could not find specified service" when nothing is loaded).
    _run_launchctl(["bootout", f"{domain}/{label}"])

    result = _run_launchctl(["bootstrap", domain, str(plist_path)])
    if result.returncode != 0:
        raise SchedulerError(
            f"launchctl bootstrap failed for {label} (exit {result.returncode}): "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    if not is_loaded(label):
        raise SchedulerError(
            f"launchctl bootstrap appeared to succeed but {label} is not loaded"
        )


def bootout_plist(label: str) -> None:
    """Remove ``label`` from the current GUI session.

    Returns silently if the job is already absent (``launchctl bootout``
    exit 36 / stderr mentioning "Could not find specified service").
    """
    domain = f"gui/{_current_uid()}"
    result = _run_launchctl(["bootout", f"{domain}/{label}"])
    if result.returncode == 0:
        return
    stderr = (result.stderr or "").lower()
    # exit 36 (ESRCH) and the textual hint both mean "wasn't loaded".
    if result.returncode == 36 or "could not find" in stderr or "no such" in stderr:
        return
    raise SchedulerError(
        f"launchctl bootout failed for {label} (exit {result.returncode}): "
        f"{(result.stderr or result.stdout or '').strip()}"
    )


def reload_schedule(
    user_config: Any, *, config_path: Path | None = None
) -> None:
    """Render + install + bootstrap both Argos launchd jobs.

    ``user_config`` must expose ``.run.time``, ``.briefing.time`` and
    ``.briefing.weekdays``. The plists are written to
    ``~/Library/LaunchAgents/com.argos.{run,brief}.plist``.

    If ``config_path`` is provided, it is plumbed through to both plist
    renderers so the scheduled jobs invoke ``argos run`` / ``argos brief``
    with ``--config <path>`` — ensuring the runtime config matches the
    one used to install the schedule.

    Translates :class:`ValueError` from malformed ``run.time`` /
    ``briefing.time`` into :class:`SchedulerError` so callers only need
    to handle one exception type.
    """
    run_time = user_config.run.time
    brief_time = user_config.briefing.time
    brief_weekdays = list(user_config.briefing.weekdays)

    # ARG-124: weekly Keep summary scheduling. The weekly knobs are optional
    # so callers passing a stripped-down SimpleNamespace (tests, legacy
    # configs) keep working — getattr falls back to spec defaults.
    weekly_enabled = bool(getattr(user_config.briefing, "weekly_enabled", True))
    weekly_time = getattr(user_config.briefing, "weekly_time", brief_time)
    weekly_weekday = getattr(user_config.briefing, "weekly_weekday", "Mon")

    # ARG-165: opt-in persistent web daemon (default False — safe for existing
    # installs that don't have `web.launchd_enabled = true` in their config).
    web_enabled = bool(getattr(getattr(user_config, "web", None), "launchd_enabled", False))

    _DEFAULT_LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_plist_path = _DEFAULT_LAUNCH_AGENTS / "com.argos.run.plist"
    brief_plist_path = _DEFAULT_LAUNCH_AGENTS / "com.argos.brief.plist"
    brief_weekly_plist_path = (
        _DEFAULT_LAUNCH_AGENTS / "com.argos.brief-weekly.plist"
    )
    web_plist_path = _DEFAULT_LAUNCH_AGENTS / "com.argos.web.plist"

    # Capture cwd once so all plists agree on the same WorkingDirectory,
    # even if some caller mutates cwd between the render calls.
    install_cwd = Path.cwd()

    try:
        run_xml = render_run_plist(
            time=run_time,
            config_path=config_path,
            working_directory=install_cwd,
        )
        brief_xml = render_brief_plist(
            time=brief_time,
            weekdays=brief_weekdays,
            config_path=config_path,
            working_directory=install_cwd,
        )
        brief_weekly_xml: str | None = None
        if weekly_enabled:
            brief_weekly_xml = render_brief_weekly_plist(
                time=weekly_time,
                weekday=weekly_weekday,
                config_path=config_path,
                working_directory=install_cwd,
            )
    except ValueError as exc:
        raise SchedulerError(f"invalid time format: {exc}") from exc

    install_plist(run_plist_path, run_xml)
    install_plist(brief_plist_path, brief_xml)
    if brief_weekly_xml is not None:
        install_plist(brief_weekly_plist_path, brief_weekly_xml)
    else:
        # Weekly disabled — best-effort bootout in case it was loaded by a
        # previous install. We don't try to delete the plist file (operators
        # might keep it for inspection) but we DO remove any loaded job.
        try:
            bootout_plist("com.argos.brief-weekly")
        except SchedulerError:
            # Tolerate cleanup failures — not enabling weekly should never
            # break daily/run scheduling.
            logger.warning(
                "weekly briefing disabled; bootout of com.argos.brief-weekly failed",
                exc_info=True,
            )

    if web_enabled:
        web_xml = render_web_plist(
            config_path=config_path,
            working_directory=install_cwd,
        )
        install_plist(web_plist_path, web_xml)
    else:
        try:
            bootout_plist("com.argos.web")
        except SchedulerError:
            logger.warning(
                "web launchd disabled; bootout of com.argos.web failed",
                exc_info=True,
            )
        # Unlike the scheduled jobs, com.argos.web is RunAtLoad + KeepAlive: a
        # plist left in ~/Library/LaunchAgents is auto-loaded at the next login
        # and would resurrect the daemon despite the opt-out. bootout only
        # clears the current session, so delete the file too — otherwise
        # disabling never sticks across reboots. install_plist regenerates it
        # whenever the operator opts back in.
        web_plist_path.unlink(missing_ok=True)

    bootstrap_plist(run_plist_path)
    try:
        bootstrap_plist(brief_plist_path)
    except SchedulerError as exc:
        raise SchedulerError(
            f"run job (com.argos.run) bootstrapped successfully but brief job "
            f"(com.argos.brief) failed: {exc}. "
            f"To clean up, run: argos schedule uninstall && argos schedule install"
        ) from exc
    if brief_weekly_xml is not None:
        try:
            bootstrap_plist(brief_weekly_plist_path)
        except SchedulerError as exc:
            raise SchedulerError(
                f"run and brief jobs bootstrapped successfully but brief-weekly "
                f"job (com.argos.brief-weekly) failed: {exc}. "
                f"To clean up, run: argos schedule uninstall && argos schedule install"
            ) from exc
    if web_enabled:
        try:
            bootstrap_plist(web_plist_path)
        except SchedulerError as exc:
            raise SchedulerError(
                f"run/brief jobs bootstrapped successfully but web daemon "
                f"(com.argos.web) failed: {exc}. "
                f"To clean up, run: argos schedule uninstall && argos schedule install"
            ) from exc


__all__ = [
    "SchedulerError",
    "bootout_plist",
    "bootstrap_plist",
    "install_plist",
    "is_loaded",
    "reload_schedule",
    "render_brief_plist",
    "render_brief_weekly_plist",
    "render_run_plist",
    "render_web_plist",
]
