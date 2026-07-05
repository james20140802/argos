"""ARG-217: end-to-end regression test for the Ollama-down run.

Simulates Ollama being unreachable during ``argos run`` and asserts the two
ARG-190 acceptance criteria together, driving the REAL wired path:

    triage (argos.brain.nodes.triage) -> run_full_pipeline Stage 6
    (argos.crawler.pipeline) -> _run (argos.cli)

Only the LLM client (triage's ``get_llm_client``) and the crawl-queue DB
helpers / session are mocked — everything in between (triage_node,
batch_triage_states, run_batch_brain_pipeline, run_full_pipeline's Stage 6
retention logic, and cli._run's exit-code check) runs unmodified. This is
what makes the test a genuine wiring regression check: if any of T1-T3's
plumbing regresses, this test fails without needing new assertions.

Reuses:
  - the ``patched_queue`` idiom from tests/crawler/test_pipeline.py (in-memory
    crawl_queue backed by _upsert_crawl_queue / _pop_from_queue / _delete_from_queue)
  - the ``_make_mock_session`` helper from tests/test_cli_run_logging.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.brain.ollama_client import OllamaInfraError
from argos.cli import _run
from argos.crawler import pipeline
from argos.models.tech_item import CategoryType


def _make_mock_session():
    """Return a mock async session context manager (see test_cli_run_logging.py)."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


class _OllamaDownClient:
    """Stand-in LLM client whose query() always fails as an infra error."""

    async def query(self, *args, **kwargs):
        raise OllamaInfraError("connection refused")

    async def unload(self, *args, **kwargs):
        return None


@pytest.fixture
def patched_queue(mocker):
    """Stub crawl-queue DB helpers so run_full_pipeline runs in-memory.

    Identical to tests/crawler/test_pipeline.py::patched_queue — duplicated
    here (rather than imported) to keep this regression test self-contained
    and immune to unrelated changes in the pipeline test module.
    """
    _stored: list[dict] = []

    async def _fake_upsert(session, items):
        _stored.clear()
        _stored.extend(items)
        return len(items)

    async def _fake_pop(session, limit):
        batch = _stored[:limit] if limit > 0 else list(_stored)
        rows = []
        for item in batch:
            row = MagicMock()
            row.source_url = item.get("source_url", "")
            row.raw_content = item.get("raw_content", "")
            row.source = item.get("_source")
            cat = item.get("_source_category")
            row.source_category = cat.value if isinstance(cat, CategoryType) else cat
            row.published_at = item.get("_published_at")
            rows.append(row)
        return rows

    delete_mock = AsyncMock()
    mocker.patch("argos.crawler.pipeline._upsert_crawl_queue", side_effect=_fake_upsert)
    mocker.patch("argos.crawler.pipeline._pop_from_queue", side_effect=_fake_pop)
    mocker.patch("argos.crawler.pipeline._delete_from_queue", new=delete_mock)
    mocker.patch("argos.crawler.pipeline._queue_count", new=AsyncMock(return_value=0))
    # Succession check is advisory and DB-backed; stub it so the AsyncMock
    # session used here doesn't trip over the savepoint-scoped call.
    mocker.patch(
        "argos.crawler.pipeline.check_succession",
        new=AsyncMock(return_value=[]),
    )
    return _stored


@pytest.mark.asyncio
async def test_ollama_down_preserves_queue_and_reports_failure(mocker, patched_queue):
    """Real triage -> Stage 6 -> _run path with Ollama down (query raises
    OllamaInfraError). Both ARG-190 acceptance criteria must hold:

    1. The seeded infra-error URLs are never passed to _delete_from_queue.
    2. _run([], ...) returns a non-zero exit code.
    """
    # 1) Ollama down: triage's LLM client raises OllamaInfraError on every query.
    mocker.patch(
        "argos.brain.nodes.triage.get_llm_client",
        lambda: _OllamaDownClient(),
    )

    # 2) Seed the in-memory crawl queue (via patched_queue) with two URLs that
    #    will hit the real crawl -> queue -> triage path.
    seeded_urls = ["https://one.example.com", "https://two.example.com"]
    crawl_items = [
        {
            "title": f"item-{i}",
            "source_url": url,
            "raw_content": f"raw content for {url}",
            "_source": "hackernews",
        }
        for i, url in enumerate(seeded_urls)
    ]
    mocker.patch(
        "argos.crawler.pipeline.run_full_crawl",
        new=AsyncMock(return_value=crawl_items),
    )

    # 3) Drive the real _run -> run_full_pipeline -> run_batch_brain_pipeline
    #    -> batch_triage_states -> triage_node chain, with only AsyncSessionLocal
    #    mocked out.
    with patch("argos.cli.AsyncSessionLocal", return_value=_make_mock_session()):
        rc = await _run([], verbose=False)

    # --- Assertion 1: queue preserved ---------------------------------------
    delete_mock = pipeline._delete_from_queue
    deleted_urls: list[str] = []
    for call in delete_mock.call_args_list:
        urls_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("urls", [])
        deleted_urls.extend(urls_arg)
    for url in seeded_urls:
        assert url not in deleted_urls, (
            f"infra-error URL {url} was deleted from crawl_queue; "
            f"Stage 6 retention regressed. deleted={deleted_urls!r}"
        )

    # --- Assertion 2: non-zero exit ------------------------------------------
    assert rc != 0, "argos run must report a non-zero exit on Ollama infra failure"
