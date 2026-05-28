from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.crawler import pipeline
from argos.crawler.pipeline import PipelineSummary
from argos.models.tech_item import CategoryType


@pytest.fixture
def patched_queue(mocker):
    """Stub crawl-queue DB helpers so run_full_pipeline tests stay in-memory.

    Items upserted via _upsert_crawl_queue are immediately returned by
    _pop_from_queue, making the queue transparent for integration tests.
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
            rows.append(row)
        return rows

    mocker.patch("argos.crawler.pipeline._upsert_crawl_queue", side_effect=_fake_upsert)
    mocker.patch("argos.crawler.pipeline._pop_from_queue", side_effect=_fake_pop)
    mocker.patch("argos.crawler.pipeline._delete_from_queue", new=AsyncMock())
    mocker.patch("argos.crawler.pipeline._queue_count", new=AsyncMock(return_value=0))
    # Stub the succession check so the bare AsyncMock() sessions used here
    # don't trip over the savepoint-scoped DB call.  Tests that exercise the
    # succession path live in test_pipeline_succession.py.
    mocker.patch(
        "argos.crawler.pipeline.check_succession",
        new=AsyncMock(return_value=[]),
    )
    return _stored


@pytest.fixture
def patched_static(mocker):
    gh = [
        {"title": "gh-1", "source_url": "https://github.com/a/b", "raw_content": "x"},
        {"title": "gh-2", "source_url": "https://github.com/c/d", "raw_content": "y"},
    ]
    hn = [
        {"title": "hn-1", "source_url": "https://hn.com/1", "raw_content": "p"},
        {"title": "hn-2", "source_url": "https://hn.com/2", "raw_content": "q"},
    ]
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(return_value=gh),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=hn),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )
    # Stub out the RSS pipeline so tests that only care about static sources
    # don't make real network requests (ARG-52).
    mocker.patch(
        "argos.crawler.pipeline.run_rss_pipeline",
        new=AsyncMock(return_value=[]),
    )
    # Stub out the arXiv pipeline so tests that only care about static sources
    # don't make real network requests (ARG-53).
    mocker.patch(
        "argos.crawler.pipeline.run_arxiv_pipeline",
        new=AsyncMock(return_value=[]),
    )
    # Stub out the SPA pipeline so tests that only care about static sources
    # don't make real network requests.
    mocker.patch(
        "argos.crawler.pipeline.run_spa_pipeline",
        new=AsyncMock(return_value=[]),
    )
    return gh, hn


async def test_run_static_pipeline_combines_sources(patched_static) -> None:
    gh, hn = patched_static
    session = AsyncMock()

    result = await pipeline.run_static_pipeline(session)

    assert len(result) == len(gh) + len(hn)
    urls = {item["source_url"] for item in result}
    for source in (*gh, *hn):
        assert source["source_url"] in urls


async def test_run_full_crawl_without_dynamic_urls_equals_static(patched_static) -> None:
    session = AsyncMock()

    static_result = await pipeline.run_static_pipeline(session)
    full_result = await pipeline.run_full_crawl(session, dynamic_urls=None)

    assert full_result == static_result


async def test_run_static_pipeline_isolates_source_failures(mocker) -> None:
    hn = [
        {"title": "hn-1", "source_url": "https://hn.com/1", "raw_content": "p"},
    ]
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(side_effect=RuntimeError("github is down")),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=hn),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    result = await pipeline.run_static_pipeline(AsyncMock())

    assert result == hn


async def test_run_full_crawl_appends_dynamic_results(patched_static, mocker) -> None:
    dynamic_item = {
        "title": "dyn",
        "source_url": "https://example.com/article",
        "raw_content": "body",
    }
    mocker.patch(
        "argos.crawler.pipeline.fetch_dynamic_page",
        new=AsyncMock(return_value=dynamic_item),
    )
    session = AsyncMock()

    result = await pipeline.run_full_crawl(
        session, dynamic_urls=["https://example.com/article"]
    )

    gh, hn = patched_static
    assert len(result) == len(gh) + len(hn) + 1
    assert result[-1] == dynamic_item


async def test_run_full_crawl_deduplicates_dynamic_against_static(patched_static, mocker) -> None:
    gh, hn = patched_static
    # dynamic URL overlaps an already-returned static URL
    duplicate_url = gh[0]["source_url"]
    duplicate_item = {"title": "dup", "source_url": duplicate_url, "raw_content": "z"}
    mocker.patch(
        "argos.crawler.pipeline.fetch_dynamic_page",
        new=AsyncMock(return_value=duplicate_item),
    )
    # Override filter_duplicate_urls to use real dedup logic (in-memory)
    async def real_dedup(_session, items):
        seen: set[str] = set()
        out = []
        for item in items:
            if item["source_url"] not in seen:
                seen.add(item["source_url"])
                out.append(item)
        return out

    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=real_dedup),
    )
    session = AsyncMock()

    result = await pipeline.run_full_crawl(session, dynamic_urls=[duplicate_url])

    urls = [item["source_url"] for item in result]
    assert urls.count(duplicate_url) == 1


async def test_run_static_pipeline_reraises_cancelled_error(mocker) -> None:
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_static_pipeline(AsyncMock())


async def test_run_full_crawl_reraises_cancelled_error_from_dynamic(patched_static, mocker) -> None:
    mocker.patch(
        "argos.crawler.pipeline.fetch_dynamic_page",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_full_crawl(
            AsyncMock(), dynamic_urls=["https://example.com/x"]
        )


async def test_run_full_crawl_skips_none_dynamic_result(patched_static, mocker) -> None:
    """fetch_dynamic_page returning None (SSRF/robots blocked) must be silently dropped."""
    mocker.patch(
        "argos.crawler.pipeline.fetch_dynamic_page",
        new=AsyncMock(return_value=None),
    )
    gh, hn = patched_static
    result = await pipeline.run_full_crawl(
        AsyncMock(), dynamic_urls=["https://example.com/blocked"]
    )
    assert len(result) == len(gh) + len(hn)


async def test_run_full_crawl_empty_list_equals_static(patched_static) -> None:
    """dynamic_urls=[] must short-circuit identically to dynamic_urls=None."""
    session = AsyncMock()
    result_none = await pipeline.run_full_crawl(session, dynamic_urls=None)
    result_empty = await pipeline.run_full_crawl(session, dynamic_urls=[])
    assert result_empty == result_none


async def test_run_static_pipeline_returns_empty_when_both_sources_fail(mocker) -> None:
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(side_effect=RuntimeError("github down")),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(side_effect=RuntimeError("hn down")),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )
    result = await pipeline.run_static_pipeline(AsyncMock())
    assert result == []


# ---------------------------------------------------------------------------
# run_full_pipeline — integration tests (brain pipeline wiring)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_full_pipeline_preflight_filter_disabled_passes_all_items(mocker, patched_queue) -> None:
    """When triage.preflight_filter is False, all items reach the batch pipeline."""
    crawl_items = [
        {"title": "job ad", "source_url": "https://a.com", "raw_content": "We're hiring"},
        {"title": "tech", "source_url": "https://b.com", "raw_content": "LangGraph 0.2"},
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[]),
    )
    # Disable the preflight filter via settings; daily_limit=0 → unlimited
    mocker.patch(
        "argos.crawler.pipeline.settings",
        **{"user.triage.preflight_filter": False, "user.run.daily_limit": 0},
    )

    _, summary = await pipeline.run_full_pipeline(AsyncMock())

    passed_items = batch_mock.call_args.args[0]
    assert len(passed_items) == 2
    assert summary.preflight_filtered == 0


@pytest.mark.asyncio
async def test_run_full_pipeline_calls_brain_for_each_item(mocker, patched_queue) -> None:
    crawl_items = [
        {"title": "t1", "source_url": "https://a.com", "raw_content": "content a"},
        {"title": "t2", "source_url": "https://b.com", "raw_content": "content b"},
    ]
    states = [
        {"is_valid": True, "source_url": "https://a.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "saved": False, "genealogy_skipped": False, "genealogy_skip_reason": None},
        {"is_valid": True, "source_url": "https://b.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "saved": False, "genealogy_skipped": False, "genealogy_skip_reason": None},
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=states),
    )

    results, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert len(results) == 2
    batch_mock.assert_called_once()
    assert isinstance(summary, PipelineSummary)
    assert summary.crawled_total == 2


@pytest.mark.asyncio
async def test_run_full_pipeline_returns_empty_on_empty_crawl(mocker, patched_queue) -> None:
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=[]))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[]),
    )
    results, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert results == []
    # batch pipeline is called with an empty list, not skipped
    batch_mock.assert_called_once_with([], mocker.ANY)
    assert isinstance(summary, PipelineSummary)
    assert summary.crawled_total == 0
    assert summary.saved_new == 0


@pytest.mark.asyncio
async def test_run_full_pipeline_skips_items_with_empty_source_url(mocker, patched_queue) -> None:
    crawl_items = [
        {"title": "no-url", "source_url": "", "raw_content": "x"},
        {"title": "has-url", "source_url": "https://good.com", "raw_content": "y"},
    ]
    good_state = {
        "is_valid": True, "source_url": "https://good.com", "raw_text": "",
        "extracted_info": None, "related_tech_ids": [], "succession_result": None,
        "saved": False, "genealogy_skipped": False, "genealogy_skip_reason": None,
    }
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[good_state]),
    )

    results, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert len(results) == 1
    # Only the item with a valid URL should reach the batch pipeline
    passed_items = batch_mock.call_args.args[0]
    assert len(passed_items) == 1
    assert passed_items[0]["source_url"] == "https://good.com"


@pytest.mark.asyncio
async def test_run_full_pipeline_continues_after_brain_failure(mocker, patched_queue) -> None:
    """Batch pipeline handling partial failures returns partial results."""
    crawl_items = [
        {"title": "item1", "source_url": "https://a.com", "raw_content": "a"},
        {"title": "item2", "source_url": "https://b.com", "raw_content": "b"},
    ]
    # Batch pipeline handles failures internally and returns what it could process
    partial_result = [
        {"is_valid": True, "source_url": "https://b.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "saved": True, "genealogy_skipped": False, "genealogy_skip_reason": None},
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=partial_result),
    )

    results, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert len(results) == 1
    assert results[0]["saved"] is True


@pytest.mark.asyncio
async def test_run_full_pipeline_summary_counts_saved_and_triage(mocker, patched_queue) -> None:
    crawl_items = [
        {"title": "t1", "source_url": "https://a.com", "raw_content": "a", "_source": "github_trending"},
        {"title": "t2", "source_url": "https://b.com", "raw_content": "b", "_source": "hackernews"},
        {"title": "t3", "source_url": "https://c.com", "raw_content": "c", "_source": "hackernews"},
    ]
    states = [
        {"is_valid": True, "saved": True, "source_url": "https://a.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "genealogy_skipped": False, "genealogy_skip_reason": None},
        {"is_valid": True, "saved": False, "source_url": "https://b.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "genealogy_skipped": False, "genealogy_skip_reason": None},
        {"is_valid": False, "saved": False, "source_url": "https://c.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "genealogy_skipped": False, "genealogy_skip_reason": None},
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=states),
    )

    results, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert summary.crawled_total == 3
    assert summary.per_source["github_trending"] == 1
    assert summary.per_source["hackernews"] == 2
    assert summary.triage_pass == 2
    assert summary.saved_new == 1
    assert summary.duration_seconds >= 0


@pytest.mark.asyncio
async def test_run_static_pipeline_tags_items_with_source(mocker) -> None:
    gh = [
        {"title": "gh-1", "source_url": "https://github.com/a/b", "raw_content": "x"},
    ]
    hn = [
        {"title": "hn-1", "source_url": "https://hn.com/1", "raw_content": "p"},
    ]
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(return_value=gh),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=hn),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    result = await pipeline.run_static_pipeline(AsyncMock())
    sources = {item["source_url"]: item.get("_source") for item in result}
    assert sources["https://github.com/a/b"] == "github_trending"
    assert sources["https://hn.com/1"] == "hackernews"


@pytest.mark.asyncio
async def test_run_full_pipeline_counts_genealogy_skipped(mocker, patched_queue) -> None:
    """summary.genealogy_skipped should equal cold-start skips; trust skips tracked separately (ARG-39, ARG-87)."""
    crawl_items = [
        {"title": "t1", "source_url": "https://a.com", "raw_content": "a", "_source": "github_trending"},
        {"title": "t2", "source_url": "https://b.com", "raw_content": "b", "_source": "hackernews"},
        {"title": "t3", "source_url": "https://c.com", "raw_content": "c", "_source": "hackernews"},
    ]
    states = [
        {"is_valid": True, "saved": True, "source_url": "https://a.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "genealogy_skipped": True, "genealogy_skip_reason": "cold_start"},
        {"is_valid": True, "saved": True, "source_url": "https://b.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "genealogy_skipped": True, "genealogy_skip_reason": "cold_start"},
        {"is_valid": True, "saved": True, "source_url": "https://c.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None,
         "genealogy_skipped": False, "genealogy_skip_reason": None},
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=states),
    )

    _, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert summary.genealogy_skipped == 2


# ---------------------------------------------------------------------------
# ARG-52: source_category forwarding from RSS items into run_brain_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_pipeline_forwards_source_category_from_rss_item(mocker, patched_queue) -> None:
    """RSS items with _source_category must be passed to run_batch_brain_pipeline
    with the key intact; GitHub/HN items without it also pass through unchanged."""
    rss_item = {
        "title": "RSS post",
        "source_url": "https://openai.com/blog/post",
        "raw_content": "rss content",
        "_source": "rss:openai.com",
        "_source_category": CategoryType.MAINSTREAM,
    }
    static_item = {
        "title": "GitHub repo",
        "source_url": "https://github.com/a/b",
        "raw_content": "github content",
        "_source": "github_trending",
    }
    crawl_items = [rss_item, static_item]

    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[]),
    )

    await pipeline.run_full_pipeline(AsyncMock())

    batch_mock.assert_called_once()
    passed_items = batch_mock.call_args.args[0]
    assert len(passed_items) == 2
    rss, static = passed_items
    assert rss["_source_category"] is CategoryType.MAINSTREAM
    assert static.get("_source_category") is None


# ---------------------------------------------------------------------------
# ARG-53: arXiv sub-pipeline integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_static_with_arxiv(mocker):
    """patched_static extended with a single arXiv item fixture."""
    arxiv_item = {
        "title": "Arxiv Paper",
        "source_url": "https://arxiv.org/abs/2401.11111",
        "raw_content": "Arxiv abstract.",
        "_source": "arxiv",
        "_source_category": CategoryType.ALPHA,
    }
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )
    mocker.patch(
        "argos.crawler.pipeline.run_rss_pipeline",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.run_arxiv_pipeline",
        new=AsyncMock(return_value=[arxiv_item]),
    )
    mocker.patch(
        "argos.crawler.pipeline.run_spa_pipeline",
        new=AsyncMock(return_value=[]),
    )
    return arxiv_item


@pytest.mark.asyncio
async def test_run_full_crawl_includes_arxiv_items(patched_static_with_arxiv) -> None:
    """arXiv items must appear in the run_full_crawl result with _source='arxiv'."""
    session = AsyncMock()
    result = await pipeline.run_full_crawl(session, dynamic_urls=None)

    assert len(result) == 1
    item = result[0]
    assert item["source_url"] == "https://arxiv.org/abs/2401.11111"
    assert item["_source"] == "arxiv"
    assert item["_source_category"] is CategoryType.ALPHA


@pytest.mark.asyncio
async def test_run_full_pipeline_forwards_source_category_from_arxiv_item(mocker, patched_queue) -> None:
    """arXiv items with _source_category must be passed to run_batch_brain_pipeline
    with the key intact so _make_initial_state can seed BrainState.source_category."""
    arxiv_item = {
        "title": "Arxiv Paper",
        "source_url": "https://arxiv.org/abs/2401.11111",
        "raw_content": "Arxiv abstract.",
        "_source": "arxiv",
        "_source_category": CategoryType.ALPHA,
    }
    mocker.patch(
        "argos.crawler.pipeline.run_full_crawl",
        new=AsyncMock(return_value=[arxiv_item]),
    )
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[]),
    )

    await pipeline.run_full_pipeline(AsyncMock())

    batch_mock.assert_called_once()
    passed_items = batch_mock.call_args.args[0]
    assert len(passed_items) == 1
    assert passed_items[0]["_source_category"] is CategoryType.ALPHA


@pytest.mark.asyncio
async def test_run_full_crawl_arxiv_failure_does_not_abort_static_rss(mocker) -> None:
    """A failing arXiv pipeline must not prevent static or RSS results from
    flowing through run_full_crawl (failure isolation parity with RSS)."""
    static_item = {
        "title": "Static Result",
        "source_url": "https://github.com/x/y",
        "raw_content": "static content",
        "_source": "github_trending",
    }
    mocker.patch(
        "argos.crawler.pipeline.run_static_pipeline",
        new=AsyncMock(return_value=[static_item]),
    )
    mocker.patch(
        "argos.crawler.pipeline.run_rss_pipeline",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.run_arxiv_pipeline",
        new=AsyncMock(side_effect=RuntimeError("arXiv API down")),
    )
    mocker.patch(
        "argos.crawler.pipeline.run_spa_pipeline",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    result = await pipeline.run_full_crawl(AsyncMock(), dynamic_urls=None)

    # Static item must still be present despite arXiv failure
    assert len(result) == 1
    assert result[0]["source_url"] == "https://github.com/x/y"


# ---------------------------------------------------------------------------
# ARG-93: crawl queue + daily_limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_pipeline_queue_selected_matches_crawled(mocker, patched_queue) -> None:
    """queue_selected in summary equals the number of items returned from queue."""
    crawl_items = [
        {"title": "t1", "source_url": "https://a.com", "raw_content": "a"},
        {"title": "t2", "source_url": "https://b.com", "raw_content": "b"},
        {"title": "t3", "source_url": "https://c.com", "raw_content": "c"},
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch("argos.crawler.pipeline.run_batch_brain_pipeline", new=AsyncMock(return_value=[]))

    _, summary = await pipeline.run_full_pipeline(AsyncMock())

    assert summary.queue_selected == 3
    assert summary.queue_remaining == 0
    assert summary.crawled_total == 3


@pytest.mark.asyncio
async def test_run_full_pipeline_daily_limit_caps_brain_items(mocker) -> None:
    """When daily_limit < total queue size, only daily_limit items reach brain pipeline."""
    crawl_items = [
        {"title": f"t{i}", "source_url": f"https://url{i}.com", "raw_content": f"c{i}"}
        for i in range(5)
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[]),
    )

    stored: list[dict] = []

    async def _fake_upsert(session, items):
        stored.clear()
        stored.extend(items)
        return len(items)

    async def _fake_pop(session, limit):
        batch = stored[:limit] if limit > 0 else list(stored)
        rows = []
        for item in batch:
            row = MagicMock()
            row.source_url = item["source_url"]
            row.raw_content = item.get("raw_content", "")
            row.source = item.get("_source")
            row.source_category = None
            rows.append(row)
        return rows

    async def _fake_count(session):
        return max(0, len(stored) - 2)

    mocker.patch("argos.crawler.pipeline._upsert_crawl_queue", side_effect=_fake_upsert)
    mocker.patch("argos.crawler.pipeline._pop_from_queue", side_effect=_fake_pop)
    mocker.patch("argos.crawler.pipeline._delete_from_queue", new=AsyncMock())
    mocker.patch("argos.crawler.pipeline._queue_count", side_effect=_fake_count)
    mocker.patch(
        "argos.crawler.pipeline.settings",
        **{"user.triage.preflight_filter": False, "user.run.daily_limit": 2},
    )

    _, summary = await pipeline.run_full_pipeline(AsyncMock())

    passed = batch_mock.call_args.args[0]
    assert len(passed) == 2
    assert summary.queue_selected == 2
    assert summary.queue_remaining == 3


@pytest.mark.asyncio
async def test_run_full_crawl_includes_spa_items(monkeypatch):
    from argos.crawler import pipeline

    monkeypatch.setattr(pipeline, "run_static_pipeline", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline, "run_rss_pipeline", AsyncMock(return_value=[]))
    monkeypatch.setattr(pipeline, "run_arxiv_pipeline", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        pipeline,
        "run_spa_pipeline",
        AsyncMock(return_value=[{"source_url": "https://anthropic.com/news/test", "raw_content": "x", "_source": "spa:anthropic"}]),
    )
    monkeypatch.setattr(pipeline, "filter_duplicate_urls", AsyncMock(side_effect=lambda s, items: items))

    mock_session = AsyncMock()
    result = await pipeline.run_full_crawl(mock_session)

    assert any("anthropic" in (r.get("_source") or "") for r in result)


@pytest.mark.asyncio
async def test_run_full_pipeline_daily_limit_zero_means_unlimited(mocker) -> None:
    """daily_limit=0 must send all queued items to brain (unlimited mode)."""
    crawl_items = [
        {"title": f"t{i}", "source_url": f"https://url{i}.com", "raw_content": f"c{i}"}
        for i in range(10)
    ]
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    batch_mock = mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline",
        new=AsyncMock(return_value=[]),
    )

    stored: list[dict] = []

    async def _fake_upsert(session, items):
        stored.clear()
        stored.extend(items)
        return len(items)

    async def _fake_pop(session, limit):
        batch = stored[:limit] if limit > 0 else list(stored)
        rows = []
        for item in batch:
            row = MagicMock()
            row.source_url = item["source_url"]
            row.raw_content = item.get("raw_content", "")
            row.source = None
            row.source_category = None
            rows.append(row)
        return rows

    mocker.patch("argos.crawler.pipeline._upsert_crawl_queue", side_effect=_fake_upsert)
    mocker.patch("argos.crawler.pipeline._pop_from_queue", side_effect=_fake_pop)
    mocker.patch("argos.crawler.pipeline._delete_from_queue", new=AsyncMock())
    mocker.patch("argos.crawler.pipeline._queue_count", new=AsyncMock(return_value=0))
    mocker.patch(
        "argos.crawler.pipeline.settings",
        **{"user.triage.preflight_filter": False, "user.run.daily_limit": 0},
    )

    _, summary = await pipeline.run_full_pipeline(AsyncMock())

    passed = batch_mock.call_args.args[0]
    assert len(passed) == 10
    assert summary.queue_selected == 10


@pytest.mark.asyncio
async def test_run_full_pipeline_forwards_published_at(mocker) -> None:
    """run_full_pipeline must include _published_at in items sent to brain pipeline."""
    from datetime import datetime, timezone

    pub = datetime(2024, 4, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Create a fake queue row with published_at set
    fake_row = MagicMock()
    fake_row.source_url = "https://example.com/article"
    fake_row.raw_content = "Some content"
    fake_row.source = "hackernews"
    fake_row.source_category = None
    fake_row.published_at = pub

    captured_items: list = []

    async def _fake_batch_brain(items, sess, **kwargs):
        captured_items.extend(items)
        return []

    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=[]))
    mocker.patch("argos.crawler.pipeline._upsert_crawl_queue", new=AsyncMock(return_value=0))
    mocker.patch(
        "argos.crawler.pipeline._pop_from_queue", new=AsyncMock(return_value=[fake_row])
    )
    mocker.patch("argos.crawler.pipeline._delete_from_queue", new=AsyncMock())
    mocker.patch("argos.crawler.pipeline._queue_count", new=AsyncMock(return_value=0))
    mocker.patch(
        "argos.crawler.pipeline.run_batch_brain_pipeline", side_effect=_fake_batch_brain
    )
    mocker.patch("argos.crawler.pipeline.check_succession", new=AsyncMock(return_value=[]))
    mocker.patch("argos.crawler.pipeline.is_preflight_reject", return_value=False)

    await pipeline.run_full_pipeline(AsyncMock())

    assert len(captured_items) == 1
    assert captured_items[0]["_published_at"] == pub
