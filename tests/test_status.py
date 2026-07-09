from __future__ import annotations

import os
import time
from pathlib import Path

import argos.status as status


RUN_LOG_SUCCESS = """\
           INFO     Saving: done (149/149)
✅ argos run 완료
─────────────────────────────
일일 처리: 150개 / 1507개 (잔여: 1357개)
신규 저장: 149개
소요 시간: 54m 30s
"""

RUN_LOG_FAILURE = """\
           INFO     Triage (8B): started
Traceback (most recent call last):
  File "x.py", line 1, in <module>
RuntimeError: boom
"""


def test_summarize_run_log_success(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(RUN_LOG_SUCCESS)
    s = status.summarize_run_log(p)
    assert s.name == "run"
    assert s.last_result == "success"
    assert s.last_success_at is not None
    assert "149" in s.detail  # 신규 저장 count surfaced


def test_summarize_run_log_failure(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(RUN_LOG_FAILURE)
    s = status.summarize_run_log(p)
    assert s.last_result == "failure"


def test_summarize_run_log_missing(tmp_path):
    s = status.summarize_run_log(tmp_path / "nope.log")
    assert s.last_result == "unknown"
    assert s.last_success_at is None
