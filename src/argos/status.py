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
_SAVED_RE = re.compile(r"신규 저장:\s*(\d+)개")
_PROCESSED_RE = re.compile(r"일일 처리:\s*([\d]+개 / [\d]+개)")
_BRIEF_SUCCESS_RE = re.compile(r"Briefing sent: ts=|No items today")
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


def summarize_run_log(path: Path, name: str = "run") -> LogSummary:
    if not path.exists():
        return LogSummary(name, "unknown", None, "로그 파일 없음")

    text = path.read_text(errors="replace")
    if _RUN_SUCCESS_MARKER in text:
        saved = _SAVED_RE.search(text)
        processed = _PROCESSED_RE.search(text)
        bits = []
        if processed:
            bits.append(f"처리 {processed.group(1)}")
        if saved:
            bits.append(f"신규 저장 {saved.group(1)}개")
        detail = ", ".join(bits) if bits else "성공"
        return LogSummary(name, "success", _mtime(path), detail)

    if "Traceback (most recent call last)" in text:
        return LogSummary(name, "failure", _mtime(path), "마지막 실행에서 예외 발생")

    return LogSummary(name, "unknown", _mtime(path), "성공/실패 마커 없음")


def summarize_brief_log(path: Path, name: str = "brief") -> LogSummary:
    if not path.exists():
        return LogSummary(name, "unknown", None, "로그 파일 없음")

    lines = path.read_text(errors="replace").splitlines()
    last_success_idx = None
    for i, line in enumerate(lines):
        if _BRIEF_SUCCESS_RE.search(line):
            last_success_idx = i

    if last_success_idx is not None:
        # Success timestamp: the ISO stamp on the marker line, else the
        # nearest preceding stamped line, else file mtime.
        ts = None
        for j in range(last_success_idx, -1, -1):
            m = _ISO_TS_RE.search(lines[j])
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                break
        return LogSummary(name, "success", ts or _mtime(path), "브리핑 발송 완료")

    if any("Traceback (most recent call last)" in ln for ln in lines):
        return LogSummary(name, "failure", _mtime(path), "마지막 실행에서 예외 발생")

    return LogSummary(name, "unknown", _mtime(path), "성공/실패 마커 없음")
