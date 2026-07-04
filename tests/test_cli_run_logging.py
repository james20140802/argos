"""Structural tests for ARG-114: shared-Console logging + progress bar fix.

These tests verify the STRUCTURAL invariants of the fix, not visual rendering:
  (a) Without -v, INFO-level records (e.g. httpx INFO) are suppressed.
  (b) With -v, INFO/DEBUG records are emitted.
  (c) The installed logging handler is a RichHandler.
  (d) The RichHandler's .console is the SAME object passed to ProgressReporter.
  (e) Progress is instantiated exactly ONCE per _run invocation.

Tests use a StringIO-backed Console, mirroring the existing test_progress.py
pattern. A fixture restores logging root state after each test to prevent
cross-test pollution.
"""

from __future__ import annotations

import io
import logging
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console
from rich.logging import RichHandler

from argos.crawler.pipeline import PipelineSummary


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_summary(**kwargs):
    defaults = {
        "crawled_total": 0,
        "per_source": {},
        "triage_pass": 0,
        "saved_new": 0,
        "genealogy_skipped": 0,
        "duration_seconds": 0.0,
    }
    defaults.update(kwargs)
    return PipelineSummary(**defaults)


def _make_mock_session():
    """Return a mock async session context manager."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.fixture(autouse=True)
def _restore_logging():
    """Snapshot and restore root logger state around each test.

    _run calls logging.basicConfig(force=True) which mutates root handlers
    and level globally. Without this fixture, logging-state leaks between
    tests.
    """
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


def _make_buf_console() -> tuple[io.StringIO, Console]:
    """Create a StringIO-backed Rich Console for test injection."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120, no_color=True)
    return buf, console


# ---------------------------------------------------------------------------
# (a) Without -v: log level depends on TTY mode
#     non-TTY → INFO  (ProgressReporter._emit lines must survive)
#     TTY     → WARNING (Rich bar handles feedback; suppress INFO noise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_default_log_level_non_tty_is_info():
    """Without verbose in non-TTY context, root level is INFO so progress logs survive."""
    buf, console = _make_buf_console()  # force_terminal=False → non-TTY

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run

        await _run([], verbose=False, console=console)

    root = logging.getLogger()
    assert root.level == logging.INFO, (
        f"Expected root log level INFO (20) for non-TTY without -v, got {root.level}"
    )


@pytest.mark.asyncio
async def test_run_default_log_level_tty_suppresses_info():
    """Without verbose in TTY context, root level is WARNING — Rich bar handles feedback."""
    buf = io.StringIO()
    tty_console = Console(file=buf, force_terminal=True, width=120, no_color=True)

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run

        await _run([], verbose=False, console=tty_console)

    root = logging.getLogger()
    assert root.level >= logging.WARNING, (
        f"Expected root log level >= WARNING (30) for TTY without -v, got {root.level}"
    )


# ---------------------------------------------------------------------------
# (b) With -v: INFO/DEBUG records are emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_verbose_log_level_passes_info():
    """With verbose=True, root logger level is DEBUG so INFO records pass through."""
    buf, console = _make_buf_console()

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run

        await _run([], verbose=True, console=console)

    root = logging.getLogger()
    assert root.level <= logging.DEBUG, (
        f"Expected root log level <= DEBUG (10) with -v, got {root.level}"
    )


# ---------------------------------------------------------------------------
# (c) Installed handler is a RichHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_installs_rich_handler():
    """_run must install a RichHandler as the root logging handler."""
    buf, console = _make_buf_console()

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run

        await _run([], verbose=False, console=console)

    root = logging.getLogger()
    rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) >= 1, (
        f"Expected at least one RichHandler on root logger, found: {root.handlers}"
    )


# ---------------------------------------------------------------------------
# (d) RichHandler.console IS the same object as ProgressReporter._console
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_rich_handler_shares_console_with_progress_reporter():
    """The RichHandler and ProgressReporter must share the SAME Console instance."""
    buf, console = _make_buf_console()
    captured_reporter: list = []

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        captured_reporter.append(progress)
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run

        await _run([], verbose=False, console=console)

    assert captured_reporter, "Pipeline was not called — test setup error"
    reporter = captured_reporter[0]

    # The reporter must expose a _console attribute set to the shared console.
    assert reporter._console is console, (
        f"ProgressReporter._console ({reporter._console!r}) is not the shared console "
        f"({console!r})"
    )

    # The RichHandler on the root logger must use the same console.
    root = logging.getLogger()
    rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert rich_handlers, "No RichHandler found on root logger"
    rh = rich_handlers[0]
    assert rh.console is console, (
        f"RichHandler.console ({rh.console!r}) is not the shared console ({console!r})"
    )


# ---------------------------------------------------------------------------
# (e) Progress is instantiated exactly ONCE per _run invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_instantiates_progress_exactly_once():
    """Rich Progress must be created exactly once per _run call (not per stage).

    Uses force_terminal=True so the ProgressReporter enters TTY (Rich) mode
    and constructs a Progress object. We verify the count is exactly 1 — not 0
    (no creation) and not >1 (stacking).
    """
    buf = io.StringIO()
    # force_terminal=True so ProgressReporter.tty=True → Progress IS created.
    tty_console = Console(file=buf, force_terminal=True, width=120, no_color=True)

    instantiation_count = {"n": 0}

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    # We patch Progress.__init__ to count instantiations.
    import rich.progress as rp

    original_init = rp.Progress.__init__

    def counting_init(self, *args, **kwargs):
        instantiation_count["n"] += 1
        original_init(self, *args, **kwargs)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
        patch.object(rp.Progress, "__init__", counting_init),
    ):
        from argos.cli import _run

        await _run([], verbose=False, console=tty_console)

    assert instantiation_count["n"] == 1, (
        f"Expected Progress to be instantiated exactly once, got {instantiation_count['n']}"
    )


# ---------------------------------------------------------------------------
# Non-regression: existing _run behavior still works (no console injected)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_works_without_injected_console():
    """_run must still work when no console is injected (production path)."""

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run

        # Must not raise; rc must be 0.
        rc = await _run([], verbose=False)
    assert rc == 0


# ---------------------------------------------------------------------------
# ARG-216: non-zero exit + warning log when triage_error is present in results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_nonzero_on_infra_error():
    """When any result carries a truthy triage_error, _run returns non-zero
    and logs a warning (Ollama-down infra failure, queue preserved for retry).

    Note: _run calls logging.basicConfig(force=True), which detaches pytest's
    caplog handler from the root logger before the warning is emitted. So we
    attach our own handler directly to the "argos.cli" logger (basicConfig
    only clears root's handlers, not named loggers') to capture the record.
    """

    async def _infra_pipeline(session, dynamic_urls=None, *, progress=None):
        results = [
            {
                "is_valid": False,
                "triage_error": "ollama down",
                "source_url": "https://a.com",
            }
        ]
        return results, _make_summary()

    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    cli_logger = logging.getLogger("argos.cli")
    handler = _ListHandler(level=logging.WARNING)
    cli_logger.addHandler(handler)
    try:
        with (
            patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
            patch("argos.cli.run_full_pipeline", new=_infra_pipeline),
        ):
            from argos.cli import _run

            rc = await _run([], verbose=False)
    finally:
        cli_logger.removeHandler(handler)

    assert rc != 0
    assert any("infra error" in r.getMessage().lower() for r in records)


@pytest.mark.asyncio
async def test_run_returns_zero_on_empty_results():
    """An empty queue (no items at all) must still return 0 — no false-red."""

    async def _empty_pipeline(session, dynamic_urls=None, *, progress=None):
        return [], _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_empty_pipeline),
    ):
        from argos.cli import _run

        rc = await _run([], verbose=False)
    assert rc == 0


@pytest.mark.asyncio
async def test_run_returns_zero_on_all_invalid_no_infra():
    """An all-invalid batch with no triage_error (e.g. malformed URLs) is not
    an infra error — must NOT be gated on triage_pass == 0, and must return 0.
    """

    async def _invalid_pipeline(session, dynamic_urls=None, *, progress=None):
        results = [{"is_valid": False, "source_url": "https://a.com"}]
        return results, _make_summary()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()),
        patch("argos.cli.run_full_pipeline", new=_invalid_pipeline),
    ):
        from argos.cli import _run

        rc = await _run([], verbose=False)
    assert rc == 0
