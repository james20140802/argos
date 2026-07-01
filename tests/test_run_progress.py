"""Tests for ARG-101: ProgressReporter integration in `argos run`.

Verifies the CLI `_run` constructs a ProgressReporter and forwards it to
`run_full_pipeline`, and that `run_full_pipeline` invokes its progress
callbacks at the expected stage boundaries.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.crawler.pipeline import PipelineSummary


_ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


def _has_ansi(text: str) -> bool:
    return bool(_ANSI_RE.search(text))


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


@pytest.mark.asyncio
async def test_run_forwards_progress_reporter_to_pipeline(monkeypatch):
    """`_run` constructs a ProgressReporter and passes it to run_full_pipeline."""
    captured = {"progress": None}

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):  # noqa: ARG001
        captured["progress"] = progress
        return [], _make_summary()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run
        rc = await _run([])

    assert rc == 0
    assert captured["progress"] is not None
    # Duck-type check: must look like a ProgressReporter (has advance / start_stage).
    assert hasattr(captured["progress"], "advance")
    assert hasattr(captured["progress"], "start_stage")


@pytest.mark.asyncio
async def test_run_non_tty_produces_no_ansi(capsys, monkeypatch):
    """Non-TTY captured output (pytest default) must contain no ANSI codes."""

    async def _fake_pipeline(session, dynamic_urls=None, *, progress=None):  # noqa: ARG001
        # Simulate a normal pipeline run so the progress reporter has work to log.
        if progress is not None:
            progress.start_stage("crawl", total=2)
            progress.advance("crawl")
            progress.advance("crawl")
            progress.finish_stage("crawl")
        return [], _make_summary(crawled_total=2)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch("argos.cli.run_full_pipeline", new=_fake_pipeline),
    ):
        from argos.cli import _run
        await _run([])

    out = capsys.readouterr().out
    assert not _has_ansi(out), f"non-TTY output should be ANSI-free, got: {out!r}"


# ---------------------------------------------------------------------------
# run_full_pipeline progress callback wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_pipeline_drives_progress_stages(monkeypatch):
    """run_full_pipeline invokes start_stage / advance on the progress object."""
    from argos.crawler import pipeline as cpipeline

    # Stub out crawler internals so we drive a fixed item set through.
    items = [
        {"source_url": f"https://e.com/{i}", "raw_content": "content", "_source": "test"}
        for i in range(3)
    ]

    async def _fake_crawl(session, dynamic_urls=None):  # noqa: ARG001
        return items

    async def _fake_upsert(session, items_in):  # noqa: ARG001
        return len(items_in)

    async def _fake_pop(session, limit):  # noqa: ARG001
        rows = []
        for i in items:
            r = MagicMock()
            r.source_url = i["source_url"]
            r.raw_content = i["raw_content"]
            r.source = i["_source"]
            r.source_category = None
            rows.append(r)
        return rows

    async def _fake_delete(session, urls):  # noqa: ARG001
        return None

    async def _fake_count(session):  # noqa: ARG001
        return 0

    monkeypatch.setattr(cpipeline, "run_full_crawl", _fake_crawl)
    monkeypatch.setattr(cpipeline, "_upsert_crawl_queue", _fake_upsert)
    monkeypatch.setattr(cpipeline, "_pop_from_queue", _fake_pop)
    monkeypatch.setattr(cpipeline, "_delete_from_queue", _fake_delete)
    monkeypatch.setattr(cpipeline, "_queue_count", _fake_count)

    async def _fake_brain(items_in, session, **kwargs):  # noqa: ARG001
        # Simulate the brain pipeline ticking via its callbacks.
        cb_t = kwargs.get("on_triage_item_done")
        cb_d = kwargs.get("on_digest_item_done")
        cb_e = kwargs.get("on_embed_item_done")
        cb_s = kwargs.get("on_save_item_done")
        for _ in items_in:
            if cb_t:
                cb_t()
            if cb_d:
                cb_d()
            if cb_e:
                cb_e()
            if cb_s:
                cb_s()
        return [
            {
                "is_valid": True,
                "saved": True,
                "source_url": i["source_url"],
                "trust_score": 0.8,
                "genealogy_skipped": False,
                "genealogy_skip_reason": None,
            }
            for i in items_in
        ]

    monkeypatch.setattr(cpipeline, "run_batch_brain_pipeline", _fake_brain)

    # Mock session with commit / flush etc.
    session = MagicMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    # Use a recording fake progress reporter.
    events: list[tuple[str, str, int | None]] = []

    class _RecordingProgress:
        def start_stage(self, name, total=None):
            events.append(("start", name, total))

        def update_total(self, name, total):
            events.append(("total", name, total))

        def advance(self, name):
            events.append(("advance", name, None))

        def finish_stage(self, name):
            events.append(("finish", name, None))

        def completed(self, name):  # pragma: no cover - unused here
            return 0

        def callback_for(self, name):
            return lambda: self.advance(name)

    progress = _RecordingProgress()
    results, summary = await cpipeline.run_full_pipeline(session, progress=progress)

    started = [e[1] for e in events if e[0] == "start"]
    assert "crawl" in started
    assert "triage" in started
    # ARG-173: digest (14B) must be wired between triage and embed so the
    # operator sees a progress bar instead of a silent gap during inference.
    assert "digest" in started
    assert started.index("triage") < started.index("digest") < started.index("embed")
    # Per spec, at least the triage / digest / save stages must tick.
    advance_stages = {e[1] for e in events if e[0] == "advance"}
    assert "triage" in advance_stages
    assert "digest" in advance_stages
    assert "save" in advance_stages
    assert summary.crawled_total == 3


@pytest.mark.asyncio
async def test_run_full_pipeline_progress_default_none_is_safe(monkeypatch):
    """When no progress reporter is passed, the old behaviour is preserved."""
    from argos.crawler import pipeline as cpipeline

    async def _fake_crawl(session, dynamic_urls=None):  # noqa: ARG001
        return []

    async def _fake_upsert(session, items_in):  # noqa: ARG001
        return 0

    async def _fake_pop(session, limit):  # noqa: ARG001
        return []

    async def _fake_delete(session, urls):  # noqa: ARG001
        return None

    async def _fake_count(session):  # noqa: ARG001
        return 0

    async def _fake_brain(items_in, session, **kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(cpipeline, "run_full_crawl", _fake_crawl)
    monkeypatch.setattr(cpipeline, "_upsert_crawl_queue", _fake_upsert)
    monkeypatch.setattr(cpipeline, "_pop_from_queue", _fake_pop)
    monkeypatch.setattr(cpipeline, "_delete_from_queue", _fake_delete)
    monkeypatch.setattr(cpipeline, "_queue_count", _fake_count)
    monkeypatch.setattr(cpipeline, "run_batch_brain_pipeline", _fake_brain)

    session = MagicMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    # Must accept no progress kwarg without raising.
    results, summary = await cpipeline.run_full_pipeline(session)
    assert results == []
    assert summary.crawled_total == 0
