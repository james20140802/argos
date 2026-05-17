"""ARG-103: run_full_pipeline calls check_succession after batch processing
and surfaces alerts via PipelineSummary.

Tests mock crawl/queue stages and run_batch_brain_pipeline; only the
succession-check wiring itself is exercised against real code.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.crawler import pipeline as crawler_pipeline
from argos.crawler.pipeline import PipelineSummary, run_full_pipeline
from argos.models.tech_succession import RelationType
from argos.slack.services.track_check import SuccessionAlert


def _saved_state(saved_item_id: uuid.UUID, source_url: str) -> dict:
    return {
        "raw_text": "x",
        "source_url": source_url,
        "is_valid": True,
        "trust_score": 0.7,
        "summary": "s",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": True,
        "saved_item_id": saved_item_id,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }


def _mk_session(queue_rows=None) -> AsyncMock:
    """Minimal AsyncSession mock with the methods run_full_pipeline pokes."""
    queue_rows = queue_rows or []
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    # session.execute is used by _upsert_crawl_queue / _pop_from_queue /
    # _delete_from_queue / _queue_count.  Each call returns a Mock whose
    # ``.scalars().all()`` / ``.scalar_one()`` is configured below.
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    pop_result = MagicMock()
    pop_result.scalars.return_value.all.return_value = queue_rows
    # Default execute behavior: return a permissive Mock so unmocked stages
    # don't crash.
    session.execute = AsyncMock(return_value=MagicMock())
    # begin_nested() returns an async context manager (SAVEPOINT).  AsyncMock
    # would return a bare coroutine for an unconfigured attribute call, which
    # breaks ``async with``.  Wire up a real async-CM object instead.
    nested_ctx = AsyncMock()
    nested_ctx.__aenter__ = AsyncMock(return_value=nested_ctx)
    nested_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_ctx)
    return session


@pytest.mark.asyncio
async def test_run_full_pipeline_scans_all_succession_rows(monkeypatch):
    """check_succession is called without a successor-ID filter so failed
    sends from prior runs can be retried (the in-query track_history
    NOT EXISTS predicate handles dedup of successful sends)."""
    saved_id_a = uuid.uuid4()
    saved_id_b = uuid.uuid4()
    user_asset_id = uuid.uuid4()

    # Stub the crawl/queue stages — we only care about the brain → succession path.
    monkeypatch.setattr(
        crawler_pipeline, "run_full_crawl", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        crawler_pipeline, "_upsert_crawl_queue", AsyncMock(return_value=0)
    )

    # Pretend two rows came out of the queue.
    queue_rows = [
        MagicMock(source_url="https://a.example", raw_content="A", source="rss", source_category=None),
        MagicMock(source_url="https://b.example", raw_content="B", source="rss", source_category=None),
    ]
    monkeypatch.setattr(
        crawler_pipeline, "_pop_from_queue", AsyncMock(return_value=queue_rows)
    )
    monkeypatch.setattr(
        crawler_pipeline, "_delete_from_queue", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        crawler_pipeline, "_queue_count", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        crawler_pipeline, "is_preflight_reject", lambda _txt: False
    )

    states = [
        _saved_state(saved_id_a, "https://a.example"),
        _saved_state(saved_id_b, "https://b.example"),
    ]
    monkeypatch.setattr(
        crawler_pipeline,
        "run_batch_brain_pipeline",
        AsyncMock(return_value=states),
    )

    expected_alert = SuccessionAlert(
        user_asset_id=user_asset_id,
        predecessor_title="Old",
        successor_title="New",
        relation_type=RelationType.REPLACE,
    )
    check_mock = AsyncMock(return_value=[expected_alert])
    monkeypatch.setattr(crawler_pipeline, "check_succession", check_mock)

    session = _mk_session(queue_rows)
    _, summary = await run_full_pipeline(session)

    # check_succession was called with the session only — no successor-ID
    # narrowing, so previously-failed alerts have a chance to re-fire.
    check_mock.assert_awaited_once()
    call = check_mock.await_args
    assert call.args[0] is session
    # Either positional only (len==1) or new_item_ids explicitly None.
    if len(call.args) > 1:
        assert call.args[1] is None
    assert call.kwargs.get("new_item_ids", None) is None

    # Alerts surface on the summary.
    assert isinstance(summary, PipelineSummary)
    assert summary.succession_alerts == [expected_alert]


@pytest.mark.asyncio
async def test_run_full_pipeline_still_checks_when_nothing_saved(monkeypatch):
    """Even with no newly-saved items, check_succession must still run so
    that any alert that failed to post on a previous run gets retried."""
    monkeypatch.setattr(
        crawler_pipeline, "run_full_crawl", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        crawler_pipeline, "_upsert_crawl_queue", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        crawler_pipeline, "_pop_from_queue", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        crawler_pipeline, "_delete_from_queue", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        crawler_pipeline, "_queue_count", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        crawler_pipeline, "run_batch_brain_pipeline", AsyncMock(return_value=[])
    )

    retried_alert = SuccessionAlert(
        user_asset_id=uuid.uuid4(),
        predecessor_title="Previously-failed predecessor",
        successor_title="Previously-failed successor",
        relation_type=RelationType.ENHANCE,
    )
    check_mock = AsyncMock(return_value=[retried_alert])
    monkeypatch.setattr(crawler_pipeline, "check_succession", check_mock)

    session = _mk_session()
    _, summary = await run_full_pipeline(session)

    # The check still runs and surfaces the retried alert.
    check_mock.assert_awaited_once()
    assert summary.succession_alerts == [retried_alert]


@pytest.mark.asyncio
async def test_run_full_pipeline_swallows_check_succession_exception(monkeypatch, caplog):
    saved_id = uuid.uuid4()

    monkeypatch.setattr(crawler_pipeline, "run_full_crawl", AsyncMock(return_value=[]))
    monkeypatch.setattr(crawler_pipeline, "_upsert_crawl_queue", AsyncMock(return_value=0))
    queue_rows = [MagicMock(source_url="https://a.example", raw_content="A", source="rss", source_category=None)]
    monkeypatch.setattr(crawler_pipeline, "_pop_from_queue", AsyncMock(return_value=queue_rows))
    monkeypatch.setattr(crawler_pipeline, "_delete_from_queue", AsyncMock(return_value=None))
    monkeypatch.setattr(crawler_pipeline, "_queue_count", AsyncMock(return_value=0))
    monkeypatch.setattr(crawler_pipeline, "is_preflight_reject", lambda _txt: False)
    monkeypatch.setattr(
        crawler_pipeline,
        "run_batch_brain_pipeline",
        AsyncMock(return_value=[_saved_state(saved_id, "https://a.example")]),
    )

    monkeypatch.setattr(
        crawler_pipeline,
        "check_succession",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    session = _mk_session(queue_rows)
    with caplog.at_level("WARNING"):
        _, summary = await run_full_pipeline(session)

    # Pipeline did not raise; alerts are empty; warning was logged.
    assert summary.succession_alerts == []
    assert any("check_succession failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_run_full_pipeline_isolates_check_succession_in_savepoint(monkeypatch):
    """When check_succession raises (typical case: a DB/statement error), the
    failure must be contained in a SAVEPOINT so the parent transaction — and
    any tech_items / tech_succession rows saved in earlier stages — stays
    intact.  Calling session.rollback() on the parent would discard the saved
    work while Stage 6 still deletes the matching queue_rows and commits,
    permanently losing the items.  Stage 6 must still run; commit must still
    succeed."""
    saved_id = uuid.uuid4()

    monkeypatch.setattr(crawler_pipeline, "run_full_crawl", AsyncMock(return_value=[]))
    monkeypatch.setattr(crawler_pipeline, "_upsert_crawl_queue", AsyncMock(return_value=0))
    queue_rows = [
        MagicMock(source_url="https://a.example", raw_content="A", source="rss", source_category=None)
    ]
    monkeypatch.setattr(crawler_pipeline, "_pop_from_queue", AsyncMock(return_value=queue_rows))
    delete_mock = AsyncMock(return_value=None)
    queue_count_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(crawler_pipeline, "_delete_from_queue", delete_mock)
    monkeypatch.setattr(crawler_pipeline, "_queue_count", queue_count_mock)
    monkeypatch.setattr(crawler_pipeline, "is_preflight_reject", lambda _txt: False)
    monkeypatch.setattr(
        crawler_pipeline,
        "run_batch_brain_pipeline",
        AsyncMock(return_value=[_saved_state(saved_id, "https://a.example")]),
    )

    monkeypatch.setattr(
        crawler_pipeline,
        "check_succession",
        AsyncMock(side_effect=RuntimeError("simulated DB statement error")),
    )

    session = _mk_session(queue_rows)
    # Track parent rollback to assert it is NOT called — that would discard
    # the items saved in earlier stages and cause the data-loss bug.
    session.rollback = AsyncMock()
    # begin_nested returns an async context manager; track that it was used.
    nested_ctx = AsyncMock()
    nested_ctx.__aenter__ = AsyncMock(return_value=nested_ctx)
    nested_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_ctx)

    _, summary = await run_full_pipeline(session)

    # Savepoint was opened around the succession check.
    session.begin_nested.assert_called_once()
    # Parent rollback was NEVER issued — saved tech_items must survive.
    session.rollback.assert_not_awaited()
    # Stage 6 still ran on the intact parent transaction.
    delete_mock.assert_awaited()
    queue_count_mock.assert_awaited()
    session.commit.assert_awaited_once()
    assert summary.succession_alerts == []


@pytest.mark.asyncio
async def test_run_full_pipeline_does_not_lose_queued_work_on_check_failure(
    monkeypatch,
):
    """Regression for the data-loss bug introduced by an unconditional
    session.rollback() on check_succession failure.

    Setup: queue row processed → saved in earlier stage → check_succession
    raises.  The processed queue row must be deleted (Stage 6) AND the parent
    transaction must commit so the saved tech_item is durable.  The previous
    fix rolled the parent transaction back, which discarded the saved item
    while Stage 6 still deleted the queue row, dropping the work permanently.
    The savepoint approach must call commit() exactly once after Stage 6 and
    must NOT issue a parent rollback that would undo the saved item.
    """
    saved_id = uuid.uuid4()

    monkeypatch.setattr(crawler_pipeline, "run_full_crawl", AsyncMock(return_value=[]))
    monkeypatch.setattr(crawler_pipeline, "_upsert_crawl_queue", AsyncMock(return_value=0))
    queue_rows = [
        MagicMock(
            source_url="https://saved.example",
            raw_content="content",
            source="rss",
            source_category=None,
        )
    ]
    monkeypatch.setattr(crawler_pipeline, "_pop_from_queue", AsyncMock(return_value=queue_rows))
    delete_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(crawler_pipeline, "_delete_from_queue", delete_mock)
    monkeypatch.setattr(crawler_pipeline, "_queue_count", AsyncMock(return_value=0))
    monkeypatch.setattr(crawler_pipeline, "is_preflight_reject", lambda _txt: False)
    monkeypatch.setattr(
        crawler_pipeline,
        "run_batch_brain_pipeline",
        AsyncMock(
            return_value=[_saved_state(saved_id, "https://saved.example")]
        ),
    )
    monkeypatch.setattr(
        crawler_pipeline,
        "check_succession",
        AsyncMock(side_effect=RuntimeError("simulated DB statement error")),
    )

    session = _mk_session(queue_rows)
    session.rollback = AsyncMock()
    nested_ctx = AsyncMock()
    nested_ctx.__aenter__ = AsyncMock(return_value=nested_ctx)
    nested_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_ctx)

    _, summary = await run_full_pipeline(session)

    # Queue cleanup ran — the saved item's source_url is removed from queue.
    delete_mock.assert_awaited_once()
    deleted_urls = delete_mock.await_args.args[1]
    assert "https://saved.example" in deleted_urls
    # Final commit ran exactly once so the saved tech_item becomes durable
    # alongside the queue cleanup.
    session.commit.assert_awaited_once()
    # Critically: parent transaction was not rolled back — that would have
    # discarded the saved tech_item even though Stage 6 deleted the queue row.
    session.rollback.assert_not_awaited()
    assert summary.succession_alerts == []


@pytest.mark.asyncio
async def test_save_node_populates_saved_item_id():
    """save_node must surface the new item's PK so the post-save hook can use it."""
    from argos.brain.nodes.save import save_node

    state = {
        "raw_text": "Title line\nbody",
        "source_url": "https://example.com/new",
        "is_valid": True,
        "trust_score": 0.5,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }

    # Mock session: no existing item, flush is a no-op.
    no_existing = MagicMock()
    no_existing.scalar_one_or_none.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=no_existing)
    session.flush = AsyncMock()

    captured: list = []

    def _capture_add(obj):
        # Stamp a deterministic UUID onto the TechItem when added so we can
        # assert against it after flush.
        if not getattr(obj, "id", None):
            obj.id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        captured.append(obj)

    session.add = _capture_add

    result = await save_node(state, session=session)

    assert result["saved"] is True
    assert result["saved_item_id"] == uuid.UUID("11111111-1111-1111-1111-111111111111")
    # save_node added one TechItem
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_save_node_does_not_set_saved_item_id_on_duplicate():
    """When the URL already exists, save_node short-circuits — no saved_item_id."""
    from argos.brain.nodes.save import save_node

    state = {
        "raw_text": "x",
        "source_url": "https://dup.example",
        "is_valid": True,
        "trust_score": 0.5,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }

    existing = MagicMock()
    existing.scalar_one_or_none.return_value = uuid.uuid4()  # row already exists
    session = AsyncMock()
    session.execute = AsyncMock(return_value=existing)
    session.flush = AsyncMock()

    result = await save_node(state, session=session)

    assert result.get("saved") is False  # untouched
    assert result.get("saved_item_id") is None


@pytest.mark.asyncio
async def test_cli_run_consumes_succession_alerts_from_summary():
    """The CLI _run helper must accept a PipelineSummary carrying alerts."""
    from argos.cli import _run

    alert = SuccessionAlert(
        user_asset_id=uuid.uuid4(),
        predecessor_title="Old",
        successor_title="New",
        relation_type=RelationType.REPLACE,
    )
    summary = PipelineSummary(crawled_total=0, succession_alerts=[alert])

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch(
            "argos.cli.run_full_pipeline",
            new=AsyncMock(return_value=([], summary)),
        ),
        # ARG-104 dispatcher patched: ARG-103 only needs to verify the wiring
        # invokes _something_ on the CLI side. We attach a sentinel callable
        # that the CLI is expected to call (or quietly skip if absent).
        patch("argos.cli._dispatch_succession_alerts", new=AsyncMock(), create=True) as dispatch_mock,
    ):
        rc = await _run([])

    assert rc == 0
    # The CLI must have attempted to dispatch alerts (the dispatcher itself is
    # implemented in ARG-104; ARG-103 only requires the wiring).
    dispatch_mock.assert_awaited_once()
    forwarded_alerts = dispatch_mock.await_args.args[0]
    assert list(forwarded_alerts) == [alert]
