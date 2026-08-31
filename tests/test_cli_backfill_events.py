import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from argos.cli import main


def test_parser_accepts_dry_run_and_limit():
    with patch("argos.cli._backfill_events", new=AsyncMock(return_value=0)) as run:
        assert main(["backfill-events", "--dry-run", "--limit", "10"]) == 0
    kwargs = run.await_args.kwargs
    assert kwargs["dry_run"] is True
    assert kwargs["limit"] == 10


def test_defaults_are_execute_mode():
    with patch("argos.cli._backfill_events", new=AsyncMock(return_value=0)) as run:
        assert main(["backfill-events"]) == 0
    kwargs = run.await_args.kwargs
    assert kwargs["dry_run"] is False
    assert kwargs["limit"] is None


def test_dry_run_report_prints_counts_and_thresholds(capsys):
    import uuid

    from argos.brain.event_backfill import Assignment, BackfillDoc, BackfillPlan
    from argos.brain.event_scoring import DocumentFeatures
    from argos.cli import _print_dry_run_report

    event_id = uuid.uuid4()
    doc = BackfillDoc(
        tech_item_id=uuid.uuid4(),
        features=DocumentFeatures(embedding=(1.0,), names=frozenset(), at=None, keywords=frozenset()),
        title="Claude 5 released",
        summary=None,
    )
    plan = BackfillPlan(assignments=[Assignment(doc=doc, event_id=event_id, created=True)])
    _print_dry_run_report(plan, total_docs=1)
    output = capsys.readouterr().out
    assert "1" in output
    assert "join_threshold" in output
    assert "window_days" in output
    assert "Claude 5 released" in output


def test_parser_accepts_batch_size():
    with patch("argos.cli._backfill_events", new=AsyncMock(return_value=0)) as run:
        assert main(["backfill-events", "--batch-size", "25"]) == 0
    kwargs = run.await_args.kwargs
    assert kwargs["batch_size"] == 25


def test_batch_size_defaults_to_50():
    with patch("argos.cli._backfill_events", new=AsyncMock(return_value=0)) as run:
        assert main(["backfill-events"]) == 0
    kwargs = run.await_args.kwargs
    assert kwargs["batch_size"] == 50


def test_execute_summary_reports_counts(capsys):
    from argos.brain.event_backfill import ExecuteResult
    from argos.cli import _print_execute_summary

    _print_execute_summary(ExecuteResult(assigned=12, created_events=5, skipped=1))
    output = capsys.readouterr().out
    assert "assigned=12" in output
    assert "new events=5" in output
    assert "skipped=1" in output


def _session_ctx():
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def test_execute_mode_exits_nonzero_when_any_document_is_skipped():
    """A skip is a swallowed exception (see execute_backfill), so it must
    surface in the exit status even when the run mostly succeeded — unlike
    backfill-digests, where a skip is a normal, always-0 outcome."""
    from argos.brain.event_backfill import BackfillDoc, ExecuteResult
    from argos.brain.event_scoring import DocumentFeatures

    doc = BackfillDoc(
        tech_item_id=uuid.uuid4(),
        features=DocumentFeatures(
            embedding=(1.0,), names=frozenset(), at=None, keywords=frozenset()
        ),
        title="t",
        summary=None,
    )
    # 9999 succeeded, 1 skipped — still must exit 1: a majority success must
    # not mask the one swallowed exception from the operator.
    fake_result = ExecuteResult(assigned=9999, created_events=1, skipped=1)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_unassigned_documents",
            AsyncMock(return_value=[doc]),
        ),
        patch(
            "argos.brain.event_backfill.execute_backfill",
            AsyncMock(return_value=fake_result),
        ),
    ):
        rc = main(["backfill-events"])

    assert rc == 1


def test_execute_mode_exits_zero_when_nothing_is_skipped():
    from argos.brain.event_backfill import BackfillDoc, ExecuteResult
    from argos.brain.event_scoring import DocumentFeatures

    doc = BackfillDoc(
        tech_item_id=uuid.uuid4(),
        features=DocumentFeatures(
            embedding=(1.0,), names=frozenset(), at=None, keywords=frozenset()
        ),
        title="t",
        summary=None,
    )
    fake_result = ExecuteResult(assigned=1, created_events=1, skipped=0)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_unassigned_documents",
            AsyncMock(return_value=[doc]),
        ),
        patch(
            "argos.brain.event_backfill.execute_backfill",
            AsyncMock(return_value=fake_result),
        ),
    ):
        rc = main(["backfill-events"])

    assert rc == 0


def test_rename_stale_flag_reaches_the_runner():
    from unittest.mock import AsyncMock, patch

    with patch("argos.cli._backfill_events", new=AsyncMock(return_value=0)) as run:
        assert main(["backfill-events", "--rename-stale"]) == 0
    assert run.await_args.kwargs["rename_stale"] is True


def test_rename_dry_run_prints_only_the_target_count(capsys):
    from argos.cli import _print_rename_dry_run

    _print_rename_dry_run(7)
    output = capsys.readouterr().out
    assert "7" in output
    assert "nothing was written" in output
