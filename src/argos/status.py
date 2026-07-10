"""Read-only summaries of the launchd run/brief logs for ``argos status``.

Parses ``~/Library/Logs/argos/{run,brief,brief-weekly}.log`` — produced by the
scheduler — into per-job summaries: last result, last success time, and a
short processed-count detail.  No DB or network access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_RUN_SUCCESS_MARKER = "✅ argos run 완료"
# A run can fail WITHOUT raising a traceback — e.g. a triage-infra error makes
# `argos run` print an explicit failure header and return non-zero (see
# cli._print_run_summary). Recognise that marker too, or such a failed run would
# masquerade as ✅ in `argos status`.
_RUN_FAILURE_MARKER = "❌ argos run 실패"
_TRACEBACK_MARKER = "Traceback (most recent call last)"
# `argos run/brief --config <path>` can exit non-zero BEFORE any run/brief work
# runs — hence before any ✅/❌ marker or traceback — when the config file is
# missing or malformed: cli._apply_config_override prints one of these
# deterministic lines to stderr and returns non-zero. The scheduled launchd job
# appends that stderr to the same append-only log, so without recognising these
# lines a failed config-load run would leave an OLDER ✅ block as the newest
# marker and `argos status` would keep reporting a stale success. (P2, PR #113
# review — same class as the non-traceback failure header above.)
_CONFIG_ERROR_RE = re.compile(
    r"^(?:Config file not found:|Invalid TOML in |Invalid config in |Could not read config file )",
    re.MULTILINE,
)
_SAVED_RE = re.compile(r"신규 저장:\s*(\d+)개")
_PROCESSED_RE = re.compile(r"일일 처리:\s*([\d]+개 / [\d]+개)")
# Matches both the daily ("Briefing sent: ts=") and weekly ("Weekly briefing
# sent: ts=") success lines from cli.py — the weekly prefix lowercases the 'b',
# so a case-insensitive 'briefing sent: ts=' covers both. Without this the
# brief-weekly.log job would render as `unknown` on every successful weekly run.
_BRIEF_SUCCESS_RE = re.compile(r"[Bb]riefing sent: ts=|No items today")
_ISO_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


@dataclass
class LogSummary:
    name: str
    last_result: str  # "success" | "failure" | "unknown"
    last_success_at: datetime | None
    detail: str


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _last_match_start(regex: re.Pattern[str], text: str) -> int:
    """Byte offset of the LAST regex match in ``text``, or -1 if none.

    Used for recency comparison against literal ``rfind`` offsets, so a
    config-load failure that lands after an older success is ranked newest.
    """
    idx = -1
    for m in regex.finditer(text):
        idx = m.start()
    return idx


def _failure_detail(path: Path) -> str:
    mtime = _mtime(path)
    if mtime is None:
        return "마지막 실행 실패"
    return f"마지막 실행 실패 ({mtime:%Y-%m-%d %H:%M})"


def summarize_run_log(path: Path, name: str = "run") -> LogSummary:
    if not path.exists():
        return LogSummary(name, "unknown", None, "로그 파일 없음")

    text = path.read_text(errors="replace")
    # Logs are append-only and never rotate, so a marker's mere *presence*
    # isn't enough — we need whichever of success/failure happened LAST.
    success_idx = text.rfind(_RUN_SUCCESS_MARKER)
    # Failure = whichever of a traceback, an explicit non-zero-exit header, OR a
    # config-load error line happened last.  The success/failure headers are
    # written to stdout by cli._print_run_summary and the config-error lines to
    # stderr by cli._apply_config_override; both land in the same append-only log,
    # so their relative order there is what decides the latest outcome.
    failure_idx = max(
        text.rfind(_TRACEBACK_MARKER),
        text.rfind(_RUN_FAILURE_MARKER),
        _last_match_start(_CONFIG_ERROR_RE, text),
    )

    if success_idx == -1 and failure_idx == -1:
        return LogSummary(name, "unknown", None, "성공/실패 마커 없음")

    if success_idx > failure_idx:
        # Scan counts only within the winning (last) success block — the file
        # may hold older run blocks whose counts must not leak into this one.
        latest = text[success_idx:]
        saved = _SAVED_RE.search(latest)
        processed = _PROCESSED_RE.search(latest)
        bits = []
        if processed:
            bits.append(f"처리 {processed.group(1)}")
        if saved:
            bits.append(f"신규 저장 {saved.group(1)}개")
        detail = ", ".join(bits) if bits else "성공"
        return LogSummary(name, "success", _mtime(path), detail)

    return LogSummary(name, "failure", None, _failure_detail(path))


def summarize_brief_log(path: Path, name: str = "brief") -> LogSummary:
    if not path.exists():
        return LogSummary(name, "unknown", None, "로그 파일 없음")

    lines = path.read_text(errors="replace").splitlines()
    # Logs are append-only and never rotate, so we need the LAST occurrence
    # of each marker, then compare positions — not just "does it exist".
    last_success_idx = None
    last_failure_idx = None
    for i, line in enumerate(lines):
        if _BRIEF_SUCCESS_RE.search(line):
            last_success_idx = i
        # A traceback OR a config-load error (from `argos brief --config` exiting
        # early via cli._apply_config_override) both mean this run failed.
        if _TRACEBACK_MARKER in line or _CONFIG_ERROR_RE.match(line):
            last_failure_idx = i

    if last_success_idx is None and last_failure_idx is None:
        return LogSummary(name, "unknown", None, "성공/실패 마커 없음")

    if last_success_idx is not None and (
        last_failure_idx is None or last_success_idx > last_failure_idx
    ):
        # Success timestamp: the ISO stamp on the marker line, else the
        # nearest preceding stamped line, else file mtime.
        ts = None
        for j in range(last_success_idx, -1, -1):
            m = _ISO_TS_RE.search(lines[j])
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                break
        return LogSummary(name, "success", ts or _mtime(path), "브리핑 발송 완료")

    return LogSummary(name, "failure", None, _failure_detail(path))


def collect_status(log_dir: Path | None = None) -> list[LogSummary]:
    from argos.scheduler import _DEFAULT_LOG_DIR

    base = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
    return [
        summarize_run_log(base / "run.log", name="run"),
        summarize_brief_log(base / "brief.log", name="brief"),
        summarize_brief_log(base / "brief-weekly.log", name="brief-weekly"),
    ]


_VERDICT_MARK = {"success": "✅", "failure": "❌", "unknown": "—"}


def render_status(summaries: list[LogSummary]) -> str:
    lines = ["", "argos status", "─" * 40]
    for s in summaries:
        mark = _VERDICT_MARK.get(s.last_result, "—")
        when = s.last_success_at.strftime("%Y-%m-%d %H:%M") if s.last_success_at else "—"
        lines.append(f"  {mark} {s.name:<13} {s.last_result:<8} 마지막 성공: {when}")
        if s.detail:
            lines.append(f"        {s.detail}")
    lines.append("")
    return "\n".join(lines)
