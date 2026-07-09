from __future__ import annotations

from datetime import datetime

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


BRIEF_LOG_SUCCESS = """\
2026-07-08 09:00:02,999 INFO httpx: HTTP Request: POST ... "HTTP/1.1 200 OK"
2026-07-09 09:00:06,663 INFO httpx: HTTP Request: POST ... "HTTP/1.1 200 OK"
Briefing sent: ts=1783555207.645869
"""

BRIEF_LOG_NO_ITEMS = "2026-07-09 09:00:06,663 INFO ...\nNo items today — briefing skipped\n"


def test_summarize_brief_log_success(tmp_path):
    p = tmp_path / "brief.log"
    p.write_text(BRIEF_LOG_SUCCESS)
    s = status.summarize_brief_log(p)
    assert s.name == "brief"
    assert s.last_result == "success"
    assert s.last_success_at == datetime(2026, 7, 9, 9, 0, 6)


def test_summarize_brief_log_no_items_is_success(tmp_path):
    p = tmp_path / "brief.log"
    p.write_text(BRIEF_LOG_NO_ITEMS)
    s = status.summarize_brief_log(p)
    assert s.last_result == "success"


def test_summarize_brief_log_missing(tmp_path):
    s = status.summarize_brief_log(tmp_path / "nope.log")
    assert s.last_result == "unknown"


def test_collect_status_reads_all_jobs(tmp_path):
    (tmp_path / "run.log").write_text(RUN_LOG_SUCCESS)
    (tmp_path / "brief.log").write_text(BRIEF_LOG_SUCCESS)
    summaries = status.collect_status(log_dir=tmp_path)
    names = {s.name for s in summaries}
    assert {"run", "brief"} <= names


def test_render_status_contains_job_names_and_verdicts(tmp_path):
    (tmp_path / "run.log").write_text(RUN_LOG_SUCCESS)
    summaries = status.collect_status(log_dir=tmp_path)
    out = status.render_status(summaries)
    assert "run" in out
    assert "success" in out or "성공" in out


def test_cmd_status_runs_and_prints(tmp_path, capsys, monkeypatch):
    import argparse

    from argos import cli

    (tmp_path / "run.log").write_text(RUN_LOG_SUCCESS)
    monkeypatch.setattr("argos.scheduler._DEFAULT_LOG_DIR", tmp_path)
    rc = cli._cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0
    assert "argos status" in out
    assert "run" in out
