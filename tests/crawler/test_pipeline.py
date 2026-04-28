from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.crawler import pipeline


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
    }
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    mocker.patch("argos.crawler.pipeline.run_brain_pipeline", new=AsyncMock(return_value=mock_state))

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    results = await pipeline.run_full_pipeline(session)
    assert len(results) == 2
    from argos.crawler import pipeline as _p
    assert _p.run_brain_pipeline.call_count == 2


@pytest.mark.asyncio
async def test_run_full_pipeline_returns_empty_on_empty_crawl(mocker) -> None:
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=[]))
    brain_mock = mocker.patch("argos.crawler.pipeline.run_brain_pipeline", new=AsyncMock())
    results = await pipeline.run_full_pipeline(AsyncMock())
    assert results == []
    brain_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_full_pipeline_skips_items_with_empty_source_url(mocker) -> None:
    crawl_items = [
        {"title": "no-url", "source_url": "", "raw_content": "x"},
        {"title": "has-url", "source_url": "https://good.com", "raw_content": "y"},
    ]
    mock_state = {
        "is_valid": True, "source_url": "https://good.com", "raw_text": "",
        "extracted_info": None, "related_tech_ids": [], "succession_result": None,
    }
    mocker.patch("argos.crawler.pipeline.run_full_crawl", new=AsyncMock(return_value=crawl_items))
    brain_mock = mocker.patch("argos.crawler.pipeline.run_brain_pipeline",
                              new=AsyncMock(return_value=mock_state))

    session = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=None)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    results = await pipeline.run_full_pipeline(session)
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

    results = await pipeline.run_full_pipeline(session)
    assert len(results) == 1
    assert call_count == 2
