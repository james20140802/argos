from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from argos.crawler import pipeline
from argos.crawler.pipeline import PipelineSummary
from argos.models.tech_item import CategoryType


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
async def test_run_full_pipeline_calls_brain_for_each_item(mocker) -> None:
    crawl_items = [
        {"title": "t1", "source_url": "https://a.com", "raw_content": "content a"},
        {"title": "t2", "source_url": "https://b.com", "raw_content": "content b"},
    ]
    mock_state = {
        "is_valid": True, "source_url": "https://a.com", "raw_text": "",
        "extracted_info": None, "related_tech_ids": [], "succession_result": None,
        "saved": False,
    }
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch("argos.crawler.pipeline.run_brain_pipeline", new=AsyncMock(return_value=mock_state))

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    results, summary = await pipeline.run_full_pipeline(session)
    assert len(results) == 2
    from argos.crawler import pipeline as _p
    assert _p.run_brain_pipeline.call_count == 2
    assert isinstance(summary, PipelineSummary)
    assert summary.crawled_total == 2


@pytest.mark.asyncio
async def test_run_full_pipeline_returns_empty_on_empty_crawl(mocker) -> None:
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=[]))
    brain_mock = mocker.patch("argos.crawler.pipeline.run_brain_pipeline", new=AsyncMock())
    results, summary = await pipeline.run_full_pipeline(AsyncMock())
    assert results == []
    brain_mock.assert_not_called()
    assert isinstance(summary, PipelineSummary)
    assert summary.crawled_total == 0
    assert summary.saved_new == 0


@pytest.mark.asyncio
async def test_run_full_pipeline_skips_items_with_empty_source_url(mocker) -> None:
    crawl_items = [
        {"title": "no-url", "source_url": "", "raw_content": "x"},
        {"title": "has-url", "source_url": "https://good.com", "raw_content": "y"},
    ]
    mock_state = {
        "is_valid": True, "source_url": "https://good.com", "raw_text": "",
        "extracted_info": None, "related_tech_ids": [], "succession_result": None,
        "saved": False,
    }
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    brain_mock = mocker.patch("argos.crawler.pipeline.run_brain_pipeline",
                              new=AsyncMock(return_value=mock_state))

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    results, summary = await pipeline.run_full_pipeline(session)
    assert len(results) == 1
    brain_mock.assert_called_once()


@pytest.mark.asyncio
async def test_run_full_pipeline_continues_after_brain_failure(mocker) -> None:
    crawl_items = [
        {"title": "bad", "source_url": "https://bad.com", "raw_content": "bad"},
        {"title": "good", "source_url": "https://good.com", "raw_content": "good"},
    ]
    good_state = {
        "is_valid": True, "source_url": "https://good.com", "raw_text": "",
        "extracted_info": None, "related_tech_ids": [], "succession_result": None,
        "saved": False,
    }
    call_count = 0

    async def brain_side_effect(raw_text, source_url, session):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("DB flush failed")
        return good_state

    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch("argos.crawler.pipeline.run_brain_pipeline",
                 new=AsyncMock(side_effect=brain_side_effect))

    class FakeNested:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False

    session = AsyncMock()
    session.begin_nested = MagicMock(return_value=FakeNested())

    results, summary = await pipeline.run_full_pipeline(session)
    assert len(results) == 1
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_full_pipeline_summary_counts_saved_and_triage(mocker) -> None:
    crawl_items = [
        {"title": "t1", "source_url": "https://a.com", "raw_content": "a", "_source": "github_trending"},
        {"title": "t2", "source_url": "https://b.com", "raw_content": "b", "_source": "hackernews"},
        {"title": "t3", "source_url": "https://c.com", "raw_content": "c", "_source": "hackernews"},
    ]
    states = [
        {"is_valid": True, "saved": True, "source_url": "https://a.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None},
        {"is_valid": True, "saved": False, "source_url": "https://b.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None},
        {"is_valid": False, "saved": False, "source_url": "https://c.com", "raw_text": "",
         "extracted_info": None, "related_tech_ids": [], "succession_result": None},
    ]
    state_iter = iter(states)
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch(
        "argos.crawler.pipeline.run_brain_pipeline",
        new=AsyncMock(side_effect=lambda **_kw: next(state_iter)),
    )

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    results, summary = await pipeline.run_full_pipeline(session)
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
async def test_run_full_pipeline_counts_genealogy_skipped(mocker) -> None:
    """summary.genealogy_skipped should equal the number of BrainStates
    whose genealogy_skipped flag is True (ARG-39)."""
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
    state_iter = iter(states)
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch(
        "argos.crawler.pipeline.run_brain_pipeline",
        new=AsyncMock(side_effect=lambda **_kw: next(state_iter)),
    )

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    _, summary = await pipeline.run_full_pipeline(session)
    assert summary.genealogy_skipped == 2


# ---------------------------------------------------------------------------
# ARG-52: source_category forwarding from RSS items into run_brain_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_pipeline_forwards_source_category_from_rss_item(mocker) -> None:
    """RSS items with _source_category must forward that hint into run_brain_pipeline
    as source_category=; GitHub/HN items without the key must call without it."""
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
        # no _source_category key
    }
    crawl_items = [rss_item, static_item]

    good_state = {
        "is_valid": True,
        "saved": False,
        "source_url": "",
        "raw_text": "",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
    }
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    brain_mock = mocker.patch(
        "argos.crawler.pipeline.run_brain_pipeline",
        new=AsyncMock(return_value=good_state),
    )

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    await pipeline.run_full_pipeline(session)

    assert brain_mock.call_count == 2
    rss_call, static_call = brain_mock.call_args_list

    # RSS item: source_category kwarg must be forwarded
    assert rss_call.kwargs.get("source_category") is CategoryType.MAINSTREAM

    # Static item: source_category kwarg must NOT be present (to avoid
    # breaking existing call signatures that don't accept the kwarg)
    assert "source_category" not in static_call.kwargs


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
async def test_run_full_pipeline_forwards_source_category_from_arxiv_item(mocker) -> None:
    """arXiv items with _source_category must forward that hint into
    run_brain_pipeline as source_category=CategoryType.ALPHA."""
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
    good_state = {
        "is_valid": True,
        "saved": False,
        "source_url": "https://arxiv.org/abs/2401.11111",
        "raw_text": "",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
    }
    brain_mock = mocker.patch(
        "argos.crawler.pipeline.run_brain_pipeline",
        new=AsyncMock(return_value=good_state),
    )

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    await pipeline.run_full_pipeline(session)

    brain_mock.assert_called_once()
    call_kwargs = brain_mock.call_args.kwargs
    assert call_kwargs.get("source_category") is CategoryType.ALPHA


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
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    result = await pipeline.run_full_crawl(AsyncMock(), dynamic_urls=None)

    # Static item must still be present despite arXiv failure
    assert len(result) == 1
    assert result[0]["source_url"] == "https://github.com/x/y"
