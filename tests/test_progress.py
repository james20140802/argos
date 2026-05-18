"""Tests for argos.progress.ProgressReporter (ARG-101).

Covers the TTY path (Rich Progress) and the non-TTY fallback (logger.info,
no ANSI escape codes).
"""

from __future__ import annotations

import io
import logging
import re
from typing import cast

from rich.console import Console

from argos.progress import ProgressReporter


# Conservative ANSI CSI matcher: ESC [ <params> <final-byte>
_ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _has_ansi(text: str) -> bool:
    return bool(_ANSI_RE.search(text))


# ---------------------------------------------------------------------------
# Non-TTY path — logger output, zero ANSI
# ---------------------------------------------------------------------------


def test_non_tty_reporter_emits_no_ansi(caplog):
    """In non-TTY mode all output goes through logger.info; no ANSI codes."""
    buf = io.StringIO()
    reporter = ProgressReporter(tty=False, file=buf)
    with caplog.at_level(logging.INFO, logger="argos.progress"):
        with reporter:
            reporter.start_stage("triage", total=3)
            reporter.advance("triage")
            reporter.advance("triage")
            reporter.advance("triage")
            reporter.finish_stage("triage")

    out = buf.getvalue()
    assert not _has_ansi(out), f"non-TTY buffer should be ANSI-free, got: {out!r}"
    # Log records should also be ANSI-free.
    for record in caplog.records:
        assert not _has_ansi(record.getMessage())


def test_non_tty_reporter_logs_stage_lifecycle(caplog):
    """Non-TTY emits at least one logger.info per start_stage and finish_stage."""
    reporter = ProgressReporter(tty=False)
    with caplog.at_level(logging.INFO, logger="argos.progress"):
        with reporter:
            reporter.start_stage("crawl", total=5)
            for _ in range(5):
                reporter.advance("crawl")
            reporter.finish_stage("crawl")

    messages = [r.getMessage() for r in caplog.records if r.name == "argos.progress"]
    joined = "\n".join(messages)
    assert "crawl" in joined.lower()
    # Final tally must mention the total.
    assert "5" in joined


def test_non_tty_advance_without_start_is_safe():
    """advance() on an unknown stage is a no-op (not an error)."""
    reporter = ProgressReporter(tty=False)
    with reporter:
        # No KeyError, no exception.
        reporter.advance("ghost")


def test_non_tty_callback_factory_ticks_advance():
    """`reporter.callback_for(name)` returns a zero-arg callable that advances."""
    reporter = ProgressReporter(tty=False)
    with reporter:
        reporter.start_stage("triage", total=3)
        cb = reporter.callback_for("triage")
        cb()
        cb()
        cb()
        assert reporter.completed("triage") == 3


def test_callback_for_default_none_is_callable_noop():
    """callback_for() always returns a callable, even before start_stage."""
    reporter = ProgressReporter(tty=False)
    with reporter:
        cb = reporter.callback_for("never_started")
        cb()  # must not raise


# ---------------------------------------------------------------------------
# TTY path — Rich Progress, ANSI codes welcome
# ---------------------------------------------------------------------------


def test_tty_reporter_writes_to_injected_console():
    """When tty=True with an injected console, Rich renders to that buffer."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, color_system="truecolor")
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("triage", total=3)
        reporter.advance("triage")
        reporter.advance("triage")
        reporter.advance("triage")
        reporter.finish_stage("triage")

    output = buf.getvalue()
    # The stage label must appear in the rendered output.
    assert "triage" in output.lower() or "triage" in _strip_ansi(output).lower()


def test_tty_reporter_advance_is_reflected_in_completed_count():
    """advance() increments the task's completed count."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("embed", total=5)
        reporter.advance("embed")
        reporter.advance("embed")
        assert reporter.completed("embed") == 2


def test_tty_finish_stage_stops_indeterminate_task():
    """finish_stage on a stage started with total=None must stop the spinner.

    Regression for PR #68 review: ``run_full_pipeline`` starts ``embed`` and
    ``genealogy`` with ``total=None``. Previously ``finish_stage`` only
    updated completion when a total was known, so the indeterminate spinner
    kept running for the rest of the progress context.
    """
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("embed", total=None)
        reporter.advance("embed")
        reporter.advance("embed")

        progress = reporter._progress
        assert progress is not None
        task_id = reporter._tasks["embed"]
        # Sanity: indeterminate task is not finished mid-flight.
        task = next(t for t in progress.tasks if t.id == task_id)
        assert task.finished is False

        reporter.finish_stage("embed")

        task = next(t for t in progress.tasks if t.id == task_id)
        assert task.finished is True, (
            "indeterminate stage should be marked finished after finish_stage"
        )


def test_tty_finish_stage_stops_indeterminate_task_with_no_advances():
    """finish_stage must stop an indeterminate stage even with zero advances."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("genealogy", total=None)
        reporter.finish_stage("genealogy")

        progress = reporter._progress
        assert progress is not None
        task_id = reporter._tasks["genealogy"]
        task = next(t for t in progress.tasks if t.id == task_id)
        assert task.finished is True


def test_tty_reporter_update_total_after_start():
    """Stage total can be updated after it was started."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("genealogy", total=None)
        reporter.update_total("genealogy", 4)
        for _ in range(4):
            reporter.advance("genealogy")
        assert reporter.completed("genealogy") == 4


# ---------------------------------------------------------------------------
# Auto-detect from sys.stdout.isatty()
# ---------------------------------------------------------------------------


def test_auto_detect_uses_isatty(monkeypatch):
    """When tty is not passed explicitly, ProgressReporter checks sys.stdout.isatty()."""
    import sys

    class _FakeStdout:
        def isatty(self):
            return False

        def write(self, _):  # pragma: no cover - never used
            return 0

        def flush(self):  # pragma: no cover - never used
            return None

    monkeypatch.setattr(sys, "stdout", cast(object, _FakeStdout()))

    reporter = ProgressReporter()
    assert reporter.tty is False


def test_callback_for_works_across_modes():
    """callback_for() returns a working callable in both tty and non-tty modes."""
    for mode in (True, False):
        buf = io.StringIO()
        if mode:
            console = Console(file=buf, force_terminal=True, width=120)
            r = ProgressReporter(tty=True, console=console)
        else:
            r = ProgressReporter(tty=False, file=buf)
        with r:
            r.start_stage("save", total=2)
            cb = r.callback_for("save")
            cb()
            cb()
            assert r.completed("save") == 2


# ---------------------------------------------------------------------------
# Reentrancy / context manager basics
# ---------------------------------------------------------------------------


def test_reporter_is_reusable_after_exit():
    """Operations outside the context are tolerated (no-op or safe)."""
    reporter = ProgressReporter(tty=False)
    with reporter:
        reporter.start_stage("crawl", total=1)
        reporter.advance("crawl")
    # After exit, further calls should not raise.
    reporter.advance("crawl")
    reporter.finish_stage("crawl")


# ---------------------------------------------------------------------------
# ARG-121: Single-bar progress invariant and log-routing regression tests
# ---------------------------------------------------------------------------


def test_single_task_per_stage_crawl():
    """Crawl stage: exactly one Rich task created regardless of advance count.

    Regression guard for ARG-114: the original bug created a new progress bar
    line on every advance/log write.  ProgressReporter must add one task per
    stage, not one per update.
    """
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("crawl", total=10)
        for _ in range(10):
            reporter.advance("crawl")
        reporter.finish_stage("crawl")

        progress = reporter._progress
        assert progress is not None
        crawl_tasks = [t for t in progress.tasks if "Crawling" in t.description]
        assert len(crawl_tasks) == 1, (
            f"Expected exactly 1 Crawling task, got {len(crawl_tasks)}"
        )


def test_single_task_per_stage_triage():
    """Triage (8B) stage: exactly one Rich task created across its lifecycle."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("triage", total=5)
        for _ in range(5):
            reporter.advance("triage")
        reporter.finish_stage("triage")

        progress = reporter._progress
        assert progress is not None
        triage_tasks = [t for t in progress.tasks if "Triage" in t.description]
        assert len(triage_tasks) == 1, (
            f"Expected exactly 1 Triage task, got {len(triage_tasks)}"
        )


def test_single_task_per_stage_embedding():
    """Embedding stage: exactly one Rich task created (indeterminate start)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    with reporter:
        reporter.start_stage("embed", total=None)
        reporter.advance("embed")
        reporter.advance("embed")
        reporter.finish_stage("embed")

        progress = reporter._progress
        assert progress is not None
        embed_tasks = [t for t in progress.tasks if "Embedding" in t.description]
        assert len(embed_tasks) == 1, (
            f"Expected exactly 1 Embedding task, got {len(embed_tasks)}"
        )


def test_multiple_stages_produce_one_task_each():
    """Running several stages in sequence yields exactly one task per stage."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    reporter = ProgressReporter(tty=True, console=console)

    stages = [
        ("crawl", 3),
        ("triage", 3),
        ("embed", None),
        ("genealogy", None),
        ("save", 3),
    ]
    with reporter:
        for name, total in stages:
            reporter.start_stage(name, total=total)
            reporter.advance(name)
            reporter.finish_stage(name)

        progress = reporter._progress
        assert progress is not None
        # Exactly as many tasks as stages — no duplicates.
        assert len(progress.tasks) == len(stages), (
            f"Expected {len(stages)} tasks, got {len(progress.tasks)}: "
            f"{[t.description for t in progress.tasks]}"
        )


def test_log_line_via_shared_console_does_not_add_progress_task():
    """Emitting a log record through a RichHandler sharing the console must
    not create a new progress task.

    Regression guard for ARG-114 problem 1: httpx INFO lines caused new bars
    to appear because logging wrote directly to stdout outside Rich's Live
    region, triggering a redraw that printed the bar again below.

    When RichHandler and ProgressReporter share the same Console, Rich routes
    the log line above the bar via the Live context — the task count stays
    constant.
    """
    from rich.logging import RichHandler

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    # Wire up the handler exactly as _configure_logging does on TTY.
    test_logger = logging.getLogger("test.arg121.shared_console")
    test_logger.setLevel(logging.DEBUG)
    handler = RichHandler(console=console, show_path=False, markup=False)
    test_logger.addHandler(handler)
    try:
        reporter = ProgressReporter(tty=True, console=console)
        with reporter:
            reporter.start_stage("crawl", total=5)
            reporter.advance("crawl")
            # Simulate an httpx-style INFO log while the bar is active.
            test_logger.info("HTTP Request: GET https://example.com 200 OK")
            reporter.advance("crawl")
            reporter.finish_stage("crawl")

            progress = reporter._progress
            assert progress is not None
            # Still exactly one task — the log did NOT add another.
            assert len(progress.tasks) == 1, (
                f"Log line added a spurious task; tasks={[t.description for t in progress.tasks]}"
            )
    finally:
        test_logger.removeHandler(handler)


def test_verbose_false_httpx_level_is_warning():
    """_configure_logging(verbose=False) clamps httpx to WARNING.

    Regression guard for ARG-118: without this helper, httpx INFO logs flood
    stdout and break the progress bar.
    """
    from argos.cli import _configure_logging

    _configure_logging(verbose=False, tty=False)
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING


def test_verbose_true_httpx_level_is_info():
    """_configure_logging(verbose=True) restores httpx to INFO.

    Regression guard for ARG-118: verbose mode must re-enable the log output
    the user explicitly requested.
    """
    from argos.cli import _configure_logging

    _configure_logging(verbose=True, tty=False)
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.INFO
