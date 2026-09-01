import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def test_rename_dry_run_makes_no_llm_call_and_prints_the_target_count(capsys):
    """Pins the guarantee that ``--rename-stale --dry-run`` never loads the
    8B model. A regression that reordered the dry-run check relative to
    ``get_llm_client()`` would only be caught by a test that drives the real
    async path — the standalone print-helper test above does not."""
    from argos.brain.event_backfill import StaleEvent
    from argos.brain.event_naming import EvidenceDoc

    events = [
        StaleEvent(event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")])
    ]
    llm_spy = MagicMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_stale_events",
            AsyncMock(return_value=events),
        ),
        patch("argos.brain.llm_client.get_llm_client", llm_spy),
    ):
        rc = main(["backfill-events", "--rename-stale", "--dry-run"])

    assert rc == 0
    llm_spy.assert_not_called()
    out = capsys.readouterr().out
    assert "1" in out
    assert "nothing was written" in out


def test_rename_stale_with_no_targets_skips_the_llm_client(capsys):
    llm_spy = MagicMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_stale_events",
            AsyncMock(return_value=[]),
        ),
        patch("argos.brain.llm_client.get_llm_client", llm_spy),
    ):
        rc = main(["backfill-events", "--rename-stale"])

    assert rc == 0
    llm_spy.assert_not_called()
    assert "no events need renaming" in capsys.readouterr().out


def test_rename_stale_unloads_the_client_after_a_normal_run():
    from argos.brain.event_backfill import RenameResult, StaleEvent
    from argos.brain.event_naming import EvidenceDoc

    events = [
        StaleEvent(event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")])
    ]
    fake_llm = AsyncMock()
    fake_llm.unload = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_stale_events",
            AsyncMock(return_value=events),
        ),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
        patch(
            "argos.brain.event_backfill.rename_stale_events",
            AsyncMock(return_value=RenameResult(renamed=1, skipped=0)),
        ),
    ):
        rc = main(["backfill-events", "--rename-stale"])

    assert rc == 0
    fake_llm.unload.assert_awaited_once_with("small")


def test_rename_stale_unloads_the_client_even_when_renaming_raises():
    """A 32B/8B swap that never unloads is the VRAM failure this codebase's
    whole model discipline exists to prevent — the ``finally`` must run even
    when ``rename_stale_events`` itself blows up, not just on its normal
    per-event failure path (which never raises out to the caller)."""
    from argos.brain.event_backfill import StaleEvent
    from argos.brain.event_naming import EvidenceDoc

    events = [
        StaleEvent(event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")])
    ]
    fake_llm = AsyncMock()
    fake_llm.unload = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_stale_events",
            AsyncMock(return_value=events),
        ),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
        patch(
            "argos.brain.event_backfill.rename_stale_events",
            AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        with pytest.raises(RuntimeError):
            main(["backfill-events", "--rename-stale"])

    fake_llm.unload.assert_awaited_once_with("small")


def test_rename_mode_exits_nonzero_when_any_event_is_skipped():
    """Mirrors ``test_execute_mode_exits_nonzero_when_any_document_is_skipped``:
    a skip in rename mode is a swallowed generation failure (see
    ``rename_stale_events``'s per-event try/except), not a benign outcome —
    an unattended run with Ollama unreachable must not exit 0."""
    from argos.brain.event_backfill import RenameResult, StaleEvent
    from argos.brain.event_naming import EvidenceDoc

    events = [
        StaleEvent(event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")])
    ]
    fake_llm = AsyncMock()
    fake_llm.unload = AsyncMock()
    # 9 succeeded, 1 skipped — still must exit 1: a majority success must
    # not mask the one swallowed exception from the operator.
    fake_result = RenameResult(renamed=9, skipped=1)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_stale_events",
            AsyncMock(return_value=events),
        ),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
        patch(
            "argos.brain.event_backfill.rename_stale_events",
            AsyncMock(return_value=fake_result),
        ),
    ):
        rc = main(["backfill-events", "--rename-stale"])

    assert rc == 1


def test_rename_mode_exits_zero_when_nothing_is_skipped():
    from argos.brain.event_backfill import RenameResult, StaleEvent
    from argos.brain.event_naming import EvidenceDoc

    events = [
        StaleEvent(event_id=uuid.uuid4(), docs=[EvidenceDoc(title="t", summary="s")])
    ]
    fake_llm = AsyncMock()
    fake_llm.unload = AsyncMock()
    fake_result = RenameResult(renamed=1, skipped=0)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=_session_ctx()),
        patch(
            "argos.brain.event_backfill.fetch_stale_events",
            AsyncMock(return_value=events),
        ),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
        patch(
            "argos.brain.event_backfill.rename_stale_events",
            AsyncMock(return_value=fake_result),
        ),
    ):
        rc = main(["backfill-events", "--rename-stale"])

    assert rc == 0


def test_dry_run_cli_path_never_calls_a_write_api(monkeypatch, capsys):
    """CLI 층까지 통틀어 --dry-run이 쓰기 API를 한 번도 부르지 않는다."""
    import asyncio
    import uuid
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock

    from argos import cli
    from argos.brain.event_backfill import BackfillDoc
    from argos.brain.event_scoring import DocumentFeatures

    session = MagicMock()

    @asynccontextmanager
    async def _cm():
        yield None

    session.begin_nested = MagicMock(side_effect=lambda: _cm())
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()

    @asynccontextmanager
    async def _session_factory():
        yield session

    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: _session_factory())
    monkeypatch.setattr(
        "argos.brain.event_backfill.fetch_unassigned_documents",
        AsyncMock(
            return_value=[
                BackfillDoc(
                    tech_item_id=uuid.uuid4(),
                    features=DocumentFeatures(
                        embedding=(1.0, 0.0),
                        names=frozenset(),
                        at=None,
                        keywords=frozenset(),
                    ),
                    title="only doc",
                    summary=None,
                )
            ]
        ),
    )

    assert asyncio.run(cli._backfill_events(dry_run=True)) == 0

    session.add.assert_not_called()
    session.add_all.assert_not_called()
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.execute.assert_not_awaited()
    assert "nothing was written" in capsys.readouterr().out
