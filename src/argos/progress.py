"""Pipeline progress reporting (ARG-92 / ARG-101).

Single ``ProgressReporter`` class that drives either a Rich
``rich.progress.Progress`` display on TTYs, or plain ``logger.info`` lines
in non-TTY environments (launchd jobs, CI logs, redirected stdout).

Design notes
------------
* The class is a context manager. Outside the context all methods become
  no-ops so callers don't have to defensively re-check state.
* Stages are named strings (``"crawl"``, ``"triage"`` …). Unknown stage
  names are accepted by ``advance()`` to keep the contract forgiving for
  the CLI (which may be plumbed into more stages than the pipeline ever
  triggers, e.g. ``genealogy`` on cold-start runs).
* TTY auto-detect uses ``sys.stdout.isatty()`` unless ``tty=`` is passed
  explicitly. Tests inject a ``rich.console.Console`` bound to a
  ``StringIO`` to deterministically exercise the Rich rendering path.
* No-op fallback: when neither TTY mode nor logger emission is desired
  callers can still construct a reporter — it remains a usable object.
"""

from __future__ import annotations

import logging
import sys
from typing import IO, Callable

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

logger = logging.getLogger(__name__)


# Pretty labels for known stages. Unknown stages fall back to the raw name
# (capitalised) so the progress bar still renders something meaningful.
_STAGE_LABELS: dict[str, str] = {
    "crawl": "Crawling",
    "preflight": "Preflight filter",
    "triage": "Triage (8B)",
    "digest": "Digest (14B)",
    "embed": "Embedding",
    "genealogy": "Genealogy (32B)",
    "save": "Saving",
}


def _label_for(name: str) -> str:
    return _STAGE_LABELS.get(name, name.replace("_", " ").capitalize())


class ProgressReporter:
    """Stage-keyed progress reporter for the ``argos run`` pipeline.

    Parameters
    ----------
    tty:
        Force TTY (Rich) or non-TTY (logger) mode. ``None`` auto-detects via
        ``sys.stdout.isatty()``.
    console:
        Optional Rich ``Console``. When provided, the TTY path renders into
        it. Tests inject a ``StringIO``-backed console; production callers
        usually leave this as ``None`` so Rich constructs its own.
    file:
        Optional file-like target for the non-TTY path. When provided, each
        ``logger.info`` line is *also* mirrored to this stream. Primarily
        for tests that want to assert on captured output without configuring
        logging handlers.
    """

    def __init__(
        self,
        tty: bool | None = None,
        *,
        console: Console | None = None,
        file: IO[str] | None = None,
    ) -> None:
        self.tty: bool = tty if tty is not None else sys.stdout.isatty()
        self._console = console
        self._file = file
        self._progress: Progress | None = None
        self._tasks: dict[str, int] = {}
        # Track completed counts ourselves so .completed() works in both
        # TTY (mirror of Rich state) and non-TTY (no Rich at all) modes.
        self._completed: dict[str, int] = {}
        self._totals: dict[str, int | None] = {}
        self._active = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ProgressReporter":
        if self.tty:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}", justify="left"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
            )
            self._progress.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._active = False
        if self._progress is not None:
            try:
                self._progress.__exit__(exc_type, exc, tb)
            finally:
                self._progress = None

    # ------------------------------------------------------------------
    # Stage API
    # ------------------------------------------------------------------

    def start_stage(self, name: str, total: int | None = None) -> None:
        """Create or restart a stage. ``total=None`` is an indeterminate bar."""
        if not self._active:
            return
        self._completed[name] = 0
        self._totals[name] = total
        if self._progress is not None:
            # Rich requires a numeric total; use 1.0 + indeterminate-style
            # description when caller hasn't pinned the total yet.
            task_total: float | None = float(total) if total is not None else None
            task_id = self._progress.add_task(
                _label_for(name),
                total=task_total,
            )
            self._tasks[name] = task_id
        else:
            self._emit(
                f"{_label_for(name)}: started"
                + (f" (total={total})" if total is not None else "")
            )

    def update_total(self, name: str, total: int) -> None:
        """Set or replace the total for a previously started stage."""
        if not self._active:
            return
        self._totals[name] = total
        if self._progress is not None:
            task_id = self._tasks.get(name)
            if task_id is not None:
                self._progress.update(task_id, total=float(total))

    def advance(self, name: str, step: int = 1) -> None:
        """Increment a stage's completed count by ``step`` (default 1)."""
        if not self._active:
            return
        # Tolerate ticks for stages that were never started (e.g. genealogy
        # on cold-start runs). Track count but don't render anything.
        self._completed[name] = self._completed.get(name, 0) + step
        if self._progress is not None:
            task_id = self._tasks.get(name)
            if task_id is not None:
                self._progress.advance(task_id, advance=step)

    def finish_stage(self, name: str) -> None:
        """Mark a stage as complete. Logs a summary line in non-TTY mode."""
        if not self._active:
            return
        if self._progress is not None:
            task_id = self._tasks.get(name)
            if task_id is not None:
                total = self._totals.get(name)
                if total is not None:
                    # Snap the bar to 100% so a stage that under-ticked still
                    # renders as completed.
                    self._progress.update(task_id, completed=float(total))
                else:
                    # Indeterminate stages (started with total=None) never get
                    # a numeric total, so the spinner would keep animating
                    # forever otherwise. Pin a total of 1 and mark it complete
                    # so Rich flips ``task.finished`` to True and stops the
                    # spinner.
                    done = self._completed.get(name, 0)
                    final = float(max(done, 1))
                    self._progress.update(
                        task_id, total=final, completed=final
                    )
                    self._progress.stop_task(task_id)
        else:
            done = self._completed.get(name, 0)
            total = self._totals.get(name)
            tally = f"{done}/{total}" if total is not None else f"{done}"
            self._emit(f"{_label_for(name)}: done ({tally})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def completed(self, name: str) -> int:
        """Return the recorded completed count for a stage (0 if unknown)."""
        return self._completed.get(name, 0)

    def callback_for(self, name: str) -> Callable[[], None]:
        """Return a zero-arg callback that ticks the given stage.

        Useful for passing into ``run_batch_brain_pipeline``'s
        ``on_*_item_done`` parameters without writing inline lambdas.
        """

        def _cb() -> None:
            self.advance(name)

        return _cb

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, message: str) -> None:
        """Emit a non-TTY status line via logger and optional file stream."""
        logger.info(message)
        if self._file is not None:
            try:
                self._file.write(message + "\n")
                self._file.flush()
            except Exception as exc:  # noqa: BLE001 - tests pass StringIO; defensive
                logger.debug("ProgressReporter file emit failed: %r", exc)
