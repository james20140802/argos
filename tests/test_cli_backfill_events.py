from unittest.mock import AsyncMock, patch

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
